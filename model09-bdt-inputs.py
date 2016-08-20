#!/usr/bin/env python

from lasagne.layers import InputLayer, DenseLayer
from lasagne.init import GlorotUniform
from lasagne.nonlinearities import rectify, sigmoid

import numpy as np
import theano.tensor as T

#----------------------------------------------------------------------

ninputs = 13


def makeModelHelper(numHiddenLayers, nodesPerHiddenLayer):

    # 2D tensor
    input_var = T.matrix('inputs')

    # 2-class problem
    noutputs = 1

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
    
    # size of minibatch
    batchSize = 32

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

            # sigmoid at output
            nonlinearity = sigmoid

            num_units = noutputs

        model = DenseLayer(model,
                           num_units = num_units,
                           W = GlorotUniform(),
                           nonlinearity = nonlinearity
                           )

    # end of loop over hidden layers

    return input_var, model

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

    assert not inputDataIsSparse, "input data is not expected to be sparse"
  
    batchSize = len(rowIndices)
  
    retval = np.zeros(batchSize, ninputs)
  
    #----------
  
    for i in range(batchSize):
  
        rowIndex = rowIndices[i]
        retval[i] = dataset.data[rowIndex]
  
    # end of loop over minibatch indices
  
    return retval

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
