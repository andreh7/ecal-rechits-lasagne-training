#!/usr/bin/env python

# modified version of train01.py for regression training of 
# the track isolation variables

import time
import numpy as np
import os, sys

import lasagne
from lasagne.objectives import squared_error, aggregate
from lasagne.updates import adam, nesterov_momentum, sgd, get_or_compute_grads

from sklearn.metrics import mean_squared_error

import theano.tensor as T
import theano

import tqdm

from Timer import Timer

from utils import sgdWithLearningRateDecay, iterate_minibatches

sys.path.append(os.path.expanduser("~/torchio")); import torchio

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

parser.add_argument('--print-model-only',
                    dest = "printModelOnlyOutput",
                    type = str,
                    default = None,
                    help='only write the model graphviz file (given after this option) and exit',
                    metavar = 'file.gv',
                    )

parser.add_argument('--monitor-gradient',
                    dest = "monitorGradient",
                    default = False,
                    action = 'store_true',
                    help='write out additional information about the gradient to the results directory',
                    )

parser.add_argument('--max-epochs',
                    dest = "maxEpochs",
                    default = None,
                    type = int,
                    help='stop after the given number of epochs',
                    )

parser.add_argument('--output-dir',
                    dest = "outputDir",
                    default = None,
                    help='manually specify the output directory',
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
print "building model"
input_vars, model = makeModel()

# produce network model in graphviz format
import draw_net
dot = draw_net.get_pydot_graph(lasagne.layers.get_all_layers(model), verbose = True)

if options.printModelOnlyOutput != None:
    # just generate the graphviz output and exit
    dot.write(options.printModelOnlyOutput, format = "raw")
    print "wrote model description to", options.printModelOnlyOutput
    sys.exit(0)

#----------
print "loading data"

cuda = True
with Timer("loading training dataset...") as t:
    trainData, trsize = datasetLoadFunction(dataDesc['train_files'], dataDesc['trsize'], 
                                            cuda = cuda,
                                            isTraining = True,
                                            reweightPtEta = False
                                            )

with Timer("loading test dataset...") as t:
    testData,  tesize = datasetLoadFunction(dataDesc['test_files'], dataDesc['tesize'], 
                                            cuda = cuda,
                                            isTraining = False,
                                            reweightPtEta = False)

# TODO: normalize these to same weight for positive and negative samples
trainWeights = trainData['weights']
testWeights  = testData['weights']

#----------
if options.outputDir == None:
    options.outputDir = "results/" + time.strftime("%Y-%m-%d-%H%M%S")

if not os.path.exists(options.outputDir):
    os.makedirs(options.outputDir)

# try to set the process name
try:
    import procname
    procname.setprocname("train " + 
                         os.path.basename(options.outputDir.rstrip('/')))
except ImportError, ex:
    pass

#----------
# write training file paths to result directory
#----------

fout = open(os.path.join(options.outputDir, "samples.txt"), "w")
for fname in dataDesc['train_files']:
    print >> fout, fname
fout.close()

#----------

logfile = open(os.path.join(options.outputDir, "train.log"), "w")

fouts = [ sys.stdout, logfile ]

#----------
# write out BDT/MVA id labels (for performance comparison)
#----------
for name, output in (
    ('train', trainData['mvaid']),
    ('test',  testData['mvaid']),
    ):
    np.savez(os.path.join(options.outputDir, "roc-data-%s-mva.npz" % name),
             # these are the BDT outputs
             output = output,
             )

# save weights (just once, we assume the ordering of the events is always the same)
np.savez(os.path.join(options.outputDir, "weights-labels-train.npz"),
         weight = trainWeights,             
         # weightBeforePtEtaReweighting = trainWeightsBeforePtEtaReweighting,
         label = trainData['labels'],
         )
np.savez(os.path.join(options.outputDir, "weights-labels-test.npz"),
         weight = testWeights,             
         label = testData['labels'],
         )


#----------
# write graphviz output to results directory
#----------
networkGraphvizFname = os.path.join(options.outputDir, "model.gv")
dot.write(networkGraphvizFname, format = "raw")

#----------

# use a 2D tensor (even if we have only one taget value)
# first index is sample index (within minibatch)
target_var = T.matrix('targets')

# these are of type theano.tensor.var.TensorVariable
train_prediction = lasagne.layers.get_output(model)
# train_loss       = squared_error(train_prediction, target_var).mean()

train_loss = aggregate(squared_error(train_prediction, target_var), mode = "mean")

# deterministic = True is e.g. set to replace dropout layers by a fixed weight
test_prediction = lasagne.layers.get_output(model, deterministic = True)
test_loss       = squared_error(test_prediction, target_var).mean()

# method for updating weights
params = lasagne.layers.get_all_params(model, trainable = True)

train_loss_grad  = theano.grad(train_loss, params)
if globals().has_key('modelParams') and modelParams.has_key('maxGradientNorm'):
    # clip the gradient, see http://lasagne.readthedocs.io/en/latest/modules/updates.html#lasagne.updates.total_norm_constraint
    train_loss_grad = lasagne.updates.total_norm_constraint(train_loss_grad, modelParams['maxGradientNorm'])


if options.optimizer == 'adam':
    updates = adam(train_loss_grad, params)
elif options.optimizer == 'sgd':

    # parameters taken from Torch examples,
    # should be equivalent
    # but this does not take into account the minibatch size 
    # (i.e. 32 times fewer evaluations of this function, learning
    # rate decays 32 times slower) ?
    updates = sgdWithLearningRateDecay(train_loss_grad, params,
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

    if options.monitorGradient:
        # see e.g. http://stackoverflow.com/a/37384861/288875
        get_train_function_grad = theano.function(input_vars + [ target_var ], train_loss_grad)

with Timer("compiling test dataset loss function...", fouts) as t:
    test_function = theano.function(input_vars + [ target_var ], test_loss)

# function to calculate the output of the network
with Timer("compiling network output function...", fouts) as t:
    test_prediction_function = theano.function(input_vars, test_prediction)

#----------
# produce test and training input once
#----------
# assuming we have enough memory 
#
# TODO: can we use slicing instead of unpacking these again for the minibatches ?
with Timer("unpacking training dataset...", fouts) as t:
    trainInput = makeInput(trainData, range(len(trainData['weights'])), inputDataIsSparse = True)
    assert len(trainInput) == len(input_vars), "number of sets of values (%d) is not equal to number of input variables (%d)" % (len(trainInput), len(input_vars))

    trainTarget = makeTarget(trainData, range(len(trainData['weights'])))

with Timer("unpacking test dataset...", fouts) as t:
    testInput  = makeInput(testData, range(len(testData['weights'])), inputDataIsSparse = True)
    testTarget = makeTarget(testData, range(len(testData['weights'])))

print "params=",params
print
print 'starting training at', time.asctime()

epoch = 1
while True:

    #----------

    if options.maxEpochs != None and epoch > options.maxEpochs:
        break

    #----------

    nowStr = time.strftime("%Y-%m-%d %H:%M:%S")
        
    for fout in fouts:

        print >> fout, "----------------------------------------"
        print >> fout, "starting epoch %d at" % epoch, nowStr
        print >> fout, "----------------------------------------"
        print >> fout, "output directory is", options.outputDir
        fout.flush()

    #----------
    # training 
    #----------

    sum_train_loss = 0
    train_batches = 0

    progbar = tqdm.tqdm(total = len(trainData['weights']), mininterval = 1.0, unit = 'samples')

    # magnitude of overall gradient. index is minibatch within epoch
    if options.monitorGradient:
        gradientMagnitudes = []

    startTime = time.time()
    for indices, targets in iterate_minibatches(trainTarget, batchsize, shuffle = True):

        # inputs = makeInput(trainData, indices, inputDataIsSparse = True)

        inputs = [ inp[indices] for inp in trainInput]

        # this also updates the weights ?
        sum_train_loss += train_function(* (inputs + [ targets ]))

        # this leads to an error
        # print train_prediction.eval()

        if options.monitorGradient:
            # this actually returns a list of CudaNdarray objects
            gradients = get_train_function_grad(* (inputs + [ targets ]))

            gradients = [ np.ndarray.flatten(np.asarray(grad)) for grad in gradients ]

            # produce the overall gradient
            gradient = np.concatenate(gradients)
            gradientMagnitudes.append(np.linalg.norm(gradient))

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
        print >> fout, "avg train loss:",sum_train_loss / float(len(trainData['weights']))
        print >> fout
        fout.flush()

    #----------
    # save gradient magnitudes
    #----------

    if options.monitorGradient:
        np.savez(os.path.join(options.outputDir, "gradient-magnitudes-%04d.npz" % epoch),
                 gradientMagnitudes = np.array(gradientMagnitudes),
                 )

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
                )[:,0]

        outputs.append(thisOutput)

    train_output, test_output = outputs
            
    # evaluating all at once exceeds the GPU memory in some cases
    # train_output = test_prediction_function(*trainInput)[:,1]
    # test_output = test_prediction_function(*testInput)[:,1]

    #----------
    # calculate RMSE
    #----------

    for name, predictions, labels, weights in  (
        ('train', train_output, trainTarget, trainData['weights']),
        ('test',  test_output,  testTarget,  testData['weights']),
        ):

        # calculate RMSE instead of AUC

        rmse = mean_squared_error(labels,
                            predictions,
                            sample_weight = weights,
                            )

        for fout in fouts:
            print >> fout
            print >> fout, "%s RMSE: %f" % (name, rmse)
            fout.flush()

        # write out online calculated rmse to the result directory
        fout = open(os.path.join(options.outputDir, "rmse-%s-%04d.txt" % (name, epoch)), "w")
        print >> fout, rmse
        fout.close()

        # write network output
        # np.savez(os.path.join(options.outputDir, "rmse-data-%s-%04d.npz" % (name, epoch)),
        #          output = predictions,
        #          )


    #----------
    # saving the model weights
    #----------

    np.savez(os.path.join(options.outputDir, 'model-%04d.npz' % epoch), 
             *lasagne.layers.get_all_param_values(model))

    #----------
    # prepare next iteration
    #----------
    epoch += 1
