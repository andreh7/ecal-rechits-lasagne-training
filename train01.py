#!/usr/bin/env python

import time
import numpy as np
import os, sys

import lasagne
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import adam, nesterov_momentum

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


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]

assert len(ARGV) == 2, "usage: " + os.path.basename(sys.argv[0]) + " modelFile.py dataFile.py"

batchsize = 32

execfile(ARGV[0])
execfile(ARGV[1])
#----------

print "loading data"

cuda = True
trainData, trsize = datasetLoadFunction(dataDesc['train_files'], dataDesc['trsize'], cuda)
testData,  tesize = datasetLoadFunction(dataDesc['test_files'], dataDesc['tesize'], cuda)

# convert labels from -1..+1 to 0..1 for cross-entropy loss
# must clone to assign

# TODO: do we still need this ?
def cloneFunc(data):
    return dict( [( key, np.copy(value) ) for key, value in data.items() ])

    ### retval = {}
    ### for key, value in data.items():
    ###     retval[key] = np.

trainData = cloneFunc(trainData); testData = cloneFunc(testData)


# TODO: normalize these to same weight for positive and negative samples
trainWeights = trainData['weights']
testWeights  = testData['weights']

#----------
print "building model"
input_vars, model = makeModel()

#----------
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

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

target_var = T.ivector('targets')
# target_var = T.vector('targets2')


train_prediction = lasagne.layers.get_output(model)
train_loss       = categorical_crossentropy(train_prediction, target_var).mean()

# deterministic = True is e.g. set to replace dropout layers by a fixed weight
test_prediction = lasagne.layers.get_output(model, deterministic = True)
test_loss       = categorical_crossentropy(test_prediction, target_var).mean()

# method for updating weights
params = lasagne.layers.get_all_params(model, trainable = True)
updates = adam(train_loss, params)
# updates = nesterov_momentum(
#           train_loss, params, learning_rate=0.01, momentum=0.9)


# build / compile the goal functions

train_function = theano.function(input_vars + [ target_var ], train_loss, updates = updates, name = 'train_function')
test_function  = theano.function(input_vars + [ target_var ], test_loss)

# function to calculate the output of the network
test_prediction_function = theano.function(input_vars, test_prediction)

#----------
# convert targets to integers (needed for softmax)
#----------

for data in (trainData, testData):
    data['labels'] = data['labels'].astype('int32')

#----------

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

        inputs = makeInput(trainData, indices, inputDataIsSparse = True)

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
    train_output = test_prediction_function(trainData['input'])
    test_output = test_prediction_function(testData['input'])

    #----------
    # calculate AUCs
    #----------

    for name, predictions, labels, weights in  (
        ('train', train_output, trainData['labels'], trainData['weights']),
        ('test',  test_output,  testData['labels'],  testData['weights']),
        ):
        auc = roc_auc_score(labels,
                            predictions[:,1],
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
                 output = predictions[:,1],
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
