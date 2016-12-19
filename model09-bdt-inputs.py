#!/usr/bin/env python

from lasagne.layers import InputLayer, DenseLayer
from lasagne.init import GlorotUniform
from lasagne.nonlinearities import rectify, sigmoid

import numpy as np
import theano.tensor as T

#----------------------------------------------------------------------

isBarrel = True

if globals().has_key('selectedVariables'):
    ninputs = len(selectedVariables)
else:
    if isBarrel:
        ninputs = 12
    else:
        ninputs = 13



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

        if i < numHiddenLayers - 1:
            # ReLU
            nonlinearity = rectify

            num_units = nodesPerHiddenLayer
        else:
            # add a dropout layer at the end ?
            #  model:add(nn.Dropout(0.3))

            # sigmoid at output: can't get this
            # to work with minibatch size > 1
            nonlinearity = sigmoid

            num_units = 1

        model = DenseLayer(model,
                           num_units = num_units,
                           W = GlorotUniform(),
                           nonlinearity = nonlinearity
                           )

    # end of loop over hidden layers

    return [ input_var ], model

#----------------------------------------------------------------------

def makeModel():
    return makeModelHelper(
        numHiddenLayers = 3,
        nodesPerHiddenLayer = ninputs * 2
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

# ----------------------------------------------------------------------
