#!/usr/bin/env python

# rechits only with multiple windows


from lasagne.layers import InputLayer, DenseLayer, ConcatLayer, Conv2DLayer, MaxPool2DLayer, ReshapeLayer
from lasagne.init import GlorotUniform
from lasagne.nonlinearities import rectify, sigmoid

import rechitmodelutils

import numpy as np
import theano.tensor as T
import math


#----------------------------------------------------------------------
# (default) model parameters
#----------------------------------------------------------------------


filterSpecs = dict(
    globalFilters = [
        # size = (ieta, iphi)
        dict(numFilters = 64, size = (5,21), name = 'supercluster'), # to pick out the supercluster.  This window can still slide a little in case of some contamination.
        dict(numFilters = 64, size = (7,23), name = 'full size'), # Full size perhaps used to measure contamination.
        dict(numFilters = 64, size = (5,5),  name = 'clean photons'),  # This is out main region for clean photons.
        dict(numFilters = 64, size = (3,3),  name = 'r9'),  # for r9...
        dict(numFilters = 64, size = (5,15)),
        dict(numFilters = 64, size = (5,9)),
        dict(numFilters = 64, size = (3,7)),
        dict(numFilters = 64, size = (3,5)),
        dict(numFilters = 64, size = (2,2),  name = 's4'),  # for S4...
        ],
    )

# size of input layer
inputLayerDimension = (7,23)

isBarrel = True

# size of minibatch
batchSize = 32

# 'reduction' factor / pool size for maxpools
poolsize = 2

#----------------------------------------------------------------------

def makeModel():

    # note that we need several input variables here
    # 3D tensor
    inputVarRecHits        = T.tensor4('rechits')

    inputLayer = InputLayer(shape=(None, 
                                   1, 
                                   inputLayerDimension[0], 
                                   inputLayerDimension[1]),
                            input_var = inputVarRecHits,
                            name = 'rechits',
                            )
    
    outputs = []
    for spec in filterSpecs['globalFilters']:

        if all([ size % 2 != 0 for size in spec['size']]):
            # all odd
            padMode = 'same'
        else:
            padMode = 'full'
        

        convOutput = Conv2DLayer(
            inputLayer, 
            num_filters = spec['numFilters'],
            filter_size = spec['size'],
            nonlinearity = rectify,
            pad = padMode,
            W = GlorotUniform(),
            name = spec.get('name', None),
            )

        poolOutput = MaxPool2DLayer(convOutput, pool_size = (poolsize, poolsize),
                                    pad = ((poolsize - 1) / 2, (poolsize - 1) / 2)
                                    )


        convOutput2 = Conv2DLayer(
            poolOutput, 
            num_filters = spec['numFilters'], 
            filter_size = (3, 3),
            nonlinearity = rectify,
            pad = 'same',
            W = GlorotUniform(),
            )

        poolOutput2 = MaxPool2DLayer(convOutput2, 
                                     pool_size = (poolsize, poolsize),
                                     pad = ((poolsize - 1) / 2, (poolsize - 1) / 2)
                                     )

        
        reshaped = ReshapeLayer(poolOutput2,
                                shape = (-1,              # minibatch dimension
                                          np.prod(poolOutput2.output_shape[1:])
                                          )
                                )

        outputs.append(reshaped)

    # end of loop over convolutional filter specifications


    # combine the outputs using a ConcatLayer
    # axis 1 (second dimension) is the layer dimension
    concatLayer = ConcatLayer(outputs,
                              axis = 1,
                              )

    network = DenseLayer(
        concatLayer,
        num_units = 2 * concatLayer.output_shape[1],
        W = GlorotUniform(),
        nonlinearity = rectify)

    # output
    network = DenseLayer(
        network,
        num_units = 1,  
        nonlinearity = sigmoid,
        W = GlorotUniform(),
        )

    return [ inputVarRecHits ], network


#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

def makeInput(dataset, rowIndices, inputDataIsSparse):
    assert inputDataIsSparse,"non-sparse input data is not supported"

    unpacker = rechitmodelutils.RecHitsUnpacker(

        inputLayerDimension[0],
        inputLayerDimension[1],

        # for shifting 18,18 (in the input) to the center
        # of our input variable ((4,12) for 7x23 input,
        # (18,18) for 35x35 input)

        recHitsXoffset = -18 + inputLayerDimension[0] / 2 + 1,
        recHitsYoffset = -18 + inputLayerDimension[1] / 2 + 1,
        )

    #----------
    # unpack the sparse data
    #----------
    recHits = unpacker.unpack(dataset, rowIndices)

    return [ recHits, 
             ]

# ----------------------------------------------------------------------
