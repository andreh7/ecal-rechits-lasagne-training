#!/usr/bin/env python

from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, prelu
from lasagne.init import GlorotUniform
from lasagne.nonlinearities import rectify, LeakyRectify, softplus, sigmoid

import numpy as np
import theano.tensor as T

#----------------------------------------------------------------------

isBarrel = True

if globals().has_key('selectedVariables') and selectedVariables != None:
    ninputs = len(selectedVariables)
else:
    if isBarrel:
        ninputs = 12
    else:
        ninputs = 13

# set to None to disable the dropout layer
dropOutProb = 0.5

# default parameters
numHiddenLayers = 3
nodesPerHiddenLayer = ninputs * 2

# put a dropout layer after each layer, not only at the end
dropOutPerLayer = False

nonlinearity = rectify

#----------------------------------------
modelParams = dict(
    # maxGradientNorm = 3.3, # typically 0.99 percentile of the gradient norm before diverging
    )

#----------------------------------------

def makeModelHelper(numHiddenLayers, nodesPerHiddenLayer):

    # 2D tensor
    input_var = T.matrix('inputs')

    # 13 input variables
    #   phoIdInput :
    #     {
    #       s4 : FloatTensor - size: 1299819
    #       scRawE : FloatTensor - size: 1299819
    #       scEta : FloatTensor - size: 1299819
    #       covIEtaIEta : FloatTensor - size: 1299819
    #       rho : FloatTensor - size: 1299819
    #       pfPhoIso03 : FloatTensor - size: 1299819
    #       phiWidth : FloatTensor - size: 1299819
    #       covIEtaIPhi : FloatTensor - size: 1299819
    #       etaWidth : FloatTensor - size: 1299819
    #       esEffSigmaRR : FloatTensor - size: 1299819
    #       r9 : FloatTensor - size: 1299819
    #       pfChgIso03 : FloatTensor - size: 1299819
    #       pfChgIso03worst : FloatTensor - size: 1299819
    #     }
    
    # how many minibatches to unpack at a time
    # and to store in the GPU (to have fewer
    # data transfers to the GPU)
    # batchesPerSuperBatch = math.floor(6636386 / batchSize)
    
    model = InputLayer(shape = (None, ninputs),
                       input_var = input_var)

    # note that the nonlinearities of the Dense
    # layer is applied on the OUTPUT side

    for i in range(numHiddenLayers):

        isLastLayer = i == numHiddenLayers - 1

        if not isLastLayer:

            thisNonlinearity = nonlinearity
            num_units = nodesPerHiddenLayer

        else:
            # sigmoid at output
            thisNonlinearity = sigmoid

            num_units = 1

        if dropOutProb != None:
            if isLastLayer or dropOutPerLayer and i > 0:
                # add a dropout layer at the end
                # or in between (but not at the beginning)
                model = DropoutLayer(model, p = dropOutProb)

        if thisNonlinearity == 'prelu':
            # special treatment for prelu layers
            # (these have a trainable parameter and
            # need a different treatment)
            model = DenseLayer(model,
                               num_units = num_units,
                               W = GlorotUniform(),
                               )
            model = prelu(model)

        else:    
            model = DenseLayer(model,
                               num_units = num_units,
                               W = GlorotUniform(),
                               nonlinearity = thisNonlinearity
                               )
            
    # end of loop over hidden layers

    return [ input_var ], model

#----------------------------------------------------------------------

def makeModel():
    return makeModelHelper(
        numHiddenLayers = numHiddenLayers,
        nodesPerHiddenLayer = nodesPerHiddenLayer
        )

#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

def makeInput(dataset, rowIndices, inputDataIsSparse):

    # assert not inputDataIsSparse, "input data is not expected to be sparse"
  
    return [ dataset['input'][rowIndices] ]

#----------------------------------------------------------------------
# function makeInputView(inputValues, first, last)
# 
#   assert(first >= 1)
#   assert(last <= inputValues:size()[1])
# 
#   return inputValues:sub(first,last)
# 
# end

#----------------------------------------------------------------------

def trainEventSelectionFunction(epoch, trainLabels, trainWeights, trainOutput):
    nevents = len(trainLabels)

    # ignore events with trainOutput above this fraction of background
    # events
    targetBgFraction = 0.2

    if epoch == 1:
        return np.arange(nevents)

    # sort background events by decreasing trainOutput
    indices = range(nevents)
    
    indices.sort(key = lambda index: trainOutput[index],
                 reverse = True
                 )


    sumBg = 0.

    totBg = trainWeights[trainLabels == 0].sum()

    maxBg = totBg * targetBgFraction

    retval = [] 

    # TODO: could use cumulative sum and filtering
    for index in indices:
        if trainLabels[index] == 0:
            # background
            sumBg += trainWeights[index]
            
        if sumBg < maxBg:
            # note that we also append this for signal
            # as long as they are in a region of high purity
            retval.append(index)

    return retval

#----------------------------------------------------------------------
