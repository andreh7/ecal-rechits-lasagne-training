#!/usr/bin/env python

import time
import numpy as np
import os, sys

import lasagne
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import adam, nesterov_momentum, sgd, get_or_compute_grads

from sklearn.metrics import roc_auc_score

import theano.tensor as T
import theano

import tqdm

sys.path.append(os.path.expanduser("~/torchio")); import torchio

#----------------------------------------------------------------------

outputDir = "results/" + time.strftime("%Y-%m-%d-%H%M%S")

#----------------------------------------------------------------------

# taken from the Lasagne mnist example and modified
def iterate_minibatches(targets, batchsize, shuffle = False):
    # generates list of indices and target values

    if shuffle:
        indices = np.arange(len(targets))
        np.random.shuffle(indices)

    for start_idx in range(0, len(targets) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield excerpt, targets[excerpt]

import time

#----------------------------------------------------------------------

# my own gradient update update supporting learning rate decay
# like we use in Torch)

# torch learning rate decay is implemented here:
# https://github.com/torch/optim/blob/b812d2a381162bed9f0df26cab8abb4015f47471/sgd.lua#L27
#
# see other Lasagne code e.g. here: https://github.com/Lasagne/Lasagne/blob/996bf64c0aec6d481044495800b461cc62040041/lasagne/updates.py#L584

def sgdWithLearningRateDecay(loss_or_grads, params, learningRate, learningRateDecay):

    from collections import OrderedDict
    from lasagne import utils

    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    t_prev = theano.shared(utils.floatX(0.))
    one = T.constant(1)

    t = t_prev + 1

    clr = learningRate / (1 + t * learningRateDecay)

    for param, grad in zip(params, grads):
        updates[param] = param - clr * grad

    updates[t_prev] = t

    return updates


#----------------------------------------------------------------------

# see http://preshing.com/20110924/timing-your-code-using-pythons-with-statement/
class Timer:    
    # timer printing a message

    def __init__(self, msg = None, fout = None):
        self.msg = msg

        if fout == None:
            fout = sys.stdout

        if isinstance(fout, list):
            # TODO: should make this for any iterable
            self.fouts = list(fout)
        else:
            self.fouts = [ fout ]


    def __enter__(self):
        if self.msg != None:
            for fout in self.fouts:
                print >> fout, self.msg,
                fout.flush()

        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        if self.msg != None:
            for fout in self.fouts:
                print >> fout, "%.1f seconds" % self.interval

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

# parse command line arguments
import argparse

parser = argparse.ArgumentParser(prog='train01.py',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                 )

parser.add_argument('--opt',
                    dest = "optimizer",
                    type = str,
                    choices = [ 'adam', 
                                'sgd',
                                ],
                    default = 'adam',
                    help='optimizer to use (default: %(default)s)'
                    )

parser.add_argument('modelFile',
                    metavar = "modelFile.py",
                    type = str,
                    nargs = 1,
                    help='file with model building code',
                    )

parser.add_argument('dataFile',
                    metavar = "dataFile.py",
                    type = str,
                    nargs = 1,
                    help='file with data loading code',
                    )

options = parser.parse_args()

#----------

batchsize = 32

execfile(options.modelFile[0])
execfile(options.dataFile[0])
#----------

print "loading data"

cuda = True
with Timer("loading training dataset...") as t:
    trainData, trsize = datasetLoadFunction(dataDesc['train_files'], dataDesc['trsize'], cuda)
with Timer("loading test dataset...") as t:
    testData,  tesize = datasetLoadFunction(dataDesc['test_files'], dataDesc['tesize'], cuda)

# convert labels from -1..+1 to 0..1 for cross-entropy loss
# must clone to assign

# TODO: normalize these to same weight for positive and negative samples
trainWeights = trainData['weights']
testWeights  = testData['weights']

#----------
print "building model"
input_vars, model = makeModel()

#----------
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

#----------
# write training file paths to result directory
#----------

fout = open(os.path.join(outputDir, "samples.txt"), "w")
for fname in dataDesc['train_files']:
    print >> fout, fname
fout.close()

#----------

logfile = open(os.path.join(outputDir, "train.log"), "w")

fouts = [ sys.stdout, logfile ]

#----------
# write out BDT/MVA id labels (for performance comparison)
#----------
for name, weights, label, output in (
    ('train', trainWeights, trainData['labels'], trainData['mvaid']),
    ('test',  testWeights,  testData['labels'],  testData['mvaid']),
    ):
    np.savez(os.path.join(outputDir, "roc-data-%s-mva.npz" % name),
             weight = weights,
             output = output,
             label = label)

#----------

### print "----------"
### print "model:"
### print model.summary()
### print "----------"
### print "the model has",model.count_params(),"parameters"
### 
### print >> logfile,"----------"
### print >> logfile,"model:"
### model.summary(file = logfile)
### print >> logfile,"----------"
### print >> logfile, "the model has",model.count_params(),"parameters"
### logfile.flush()

import draw_net
networkGraphvizFname = os.path.join(outputDir, "model.gv")

dot = draw_net.get_pydot_graph(lasagne.layers.get_all_layers(model), verbose = True)
dot.write(networkGraphvizFname, format = "raw")

# runs dot externally but graphviz is not installed on the machines...
if False:
    for suffix in ("svg",):
        draw_net.draw_to_file(lasagne.layers.get_all_layers(model), 
                              os.path.join(outputDir, "model." + suffix))

#----------

target_var = T.ivector('targets')
# target_var = T.vector('targets2')

# these are of type theano.tensor.var.TensorVariable
train_prediction = lasagne.layers.get_output(model)
train_loss       = categorical_crossentropy(train_prediction, target_var).mean()

# deterministic = True is e.g. set to replace dropout layers by a fixed weight
test_prediction = lasagne.layers.get_output(model, deterministic = True)
test_loss       = categorical_crossentropy(test_prediction, target_var).mean()

# method for updating weights
params = lasagne.layers.get_all_params(model, trainable = True)


if options.optimizer == 'adam':
    updates = adam(train_loss, params)
elif options.optimizer == 'sgd':

    # parameters taken from Torch examples,
    # should be equivalent
    # but this does not take into account the minibatch size 
    # (i.e. 32 times fewer evaluations of this function, learning
    # rate decays 32 times slower) ?
    updates = sgdWithLearningRateDecay(train_loss, params,
                                       learningRate = 1e-3,
                                       learningRateDecay = 1e-7)

    # original torch parameters:                 
    # optimState = {
    #    -- learning rate at beginning
    #    learningRate = 1e-3,
    #    weightDecay = 0,
    #    momentum = 0,
    #    learningRateDecay = 1e-7
    # }


else:
    raise Exception("internal error")

# updates = nesterov_momentum(
#           train_loss, params, learning_rate=0.01, momentum=0.9)


#----------
# build / compile the goal functions
#----------

with Timer("compiling train dataset loss function...", fouts) as t:
    train_function = theano.function(input_vars + [ target_var ], train_loss, updates = updates, name = 'train_function')

with Timer("compiling test dataset loss function...", fouts) as t:
    test_function  = theano.function(input_vars + [ target_var ], test_loss)

# function to calculate the output of the network
with Timer("compiling network output function...", fouts) as t:
    test_prediction_function = theano.function(input_vars, test_prediction)

#----------
# convert targets to integers (needed for softmax)
#----------

for data in (trainData, testData):
    data['labels'] = data['labels'].astype('int32')

#----------
# produce test and training input once
#----------
# assuming we have enough memory 
#
# TODO: can we use slicing instead of unpacking these again for the minibatches ?
with Timer("unpacking training dataset...", fouts) as t:
    trainInput = makeInput(trainData, range(len(trainData['labels'])), inputDataIsSparse = True)
    assert len(trainInput) == len(input_vars), "number of sets of values (%d) is not equal to number of input variables (%d)" % (len(trainInput), len(input_vars))

with Timer("unpacking test dataset...", fouts) as t:
    testInput  = makeInput(testData, range(len(testData['labels'])), inputDataIsSparse = True)

print "params=",params
print
print 'starting training at', time.asctime()

epoch = 1
while True:

    nowStr = time.strftime("%Y-%m-%d %H:%M:%S")
        
    for fout in fouts:

        print >> fout, "----------------------------------------"
        print >> fout, "starting epoch %d at" % epoch, nowStr
        print >> fout, "----------------------------------------"
        print >> fout, "output directory is", outputDir
        fout.flush()

    #----------
    # training 
    #----------

    sum_train_loss = 0
    train_batches = 0

    progbar = tqdm.tqdm(total = len(trainData['labels']), mininterval = 1.0, unit = 'samples')

    startTime = time.time()
    for indices, targets in iterate_minibatches(trainData['labels'], batchsize, shuffle = True):

        # inputs = makeInput(trainData, indices, inputDataIsSparse = True)

        inputs = [ inp[indices] for inp in trainInput]

        # this also updates the weights ?
        sum_train_loss += train_function(* (inputs + [ targets ]))

        # this leads to an error
        # print train_prediction.eval()

        train_batches += 1

        progbar.update(batchsize)

    # end of loop over minibatches
    progbar.close()

    #----------

    deltaT = time.time() - startTime

    for fout in fouts:
        print >> fout
        print >> fout, "time to learn 1 sample: %.3f ms" % ( deltaT / len(trainWeights) * 1000.0)
        print >> fout, "time to train entire batch: %.2f min" % (deltaT / 60.0)
        print >> fout
        print >> fout, "avg train loss:",sum_train_loss / float(len(trainData['labels']))
        print >> fout
        fout.flush()

    #----------
    # calculate outputs of train and test samples
    #----------

    evalBatchSize = 10000

    outputs = []

    for input in (trainInput, testInput):
        numSamples = input[0].shape[0]
        
        thisOutput = np.zeros(numSamples)

        for start in range(0,numSamples,evalBatchSize):
            end = min(start + evalBatchSize,numSamples)

            thisOutput[start:end] = test_prediction_function(
                *[ inp[start:end] for inp in input]
                )[:,1]

        outputs.append(thisOutput)

    train_output, test_output = outputs
            
    # evaluating all at once exceeds the GPU memory in some cases
    # train_output = test_prediction_function(*trainInput)[:,1]
    # test_output = test_prediction_function(*testInput)[:,1]

    #----------
    # calculate AUCs
    #----------

    for name, predictions, labels, weights in  (
        ('train', train_output, trainData['labels'], trainData['weights']),
        ('test',  test_output,  testData['labels'],  testData['weights']),
        ):
        auc = roc_auc_score(labels,
                            predictions,
                            sample_weight = weights,
                            average = None,
                            )

        for fout in fouts:
            print >> fout
            print >> fout, "%s AUC: %f" % (name, auc)
            fout.flush()

        # write network output
        np.savez(os.path.join(outputDir, "roc-data-%s-%04d.npz" % (name, epoch)),
                 weight = weights,
                 output = predictions,
                 label = labels)


    #----------
    # saving the model weights
    #----------

    np.savez(os.path.join(outputDir, 'model-%04d.npz' % epoch), 
             *lasagne.layers.get_all_param_values(model))

    #----------
    # prepare next iteration
    #----------
    epoch += 1
