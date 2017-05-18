#!/usr/bin/env python

# rechits only, with multiple convolutional filter sizes
# wide (35x36) rechit inputs

#----------------------------------------------------------------------
# (default) model parameters
#----------------------------------------------------------------------


filterSpecs = dict(
    globalFilters = [
        # size = (ieta, iphi)
        dict(numFilters = 32, size = (3,3)),
        dict(numFilters = 32, size = (5,5)),
        
        dict(numFilters = 32, size = (7,11)),
        
        dict(numFilters = 32, size = (7,23)),
        ],

    outerFilters = [
        dict(numFilters = 32, size = (3,3)),
        dict(numFilters = 32, size = (5,5)),
        
        dict(numFilters = 32, size = (7,11)),
        
        dict(numFilters = 32, size = (7,23)),
        ],

    # hole of rechits ignored for the outer filters
    innerHoleSize = (5,5),

    )

# size of input layer
inputLayerDimension = (35,35)

isBarrel = True

# size of minibatch
batchSize = 32

#----------------------------------------------------------------------

from lasagne.layers import InputLayer, DenseLayer, ConcatLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer, ReshapeLayer
from lasagne.init import GlorotUniform
from lasagne.nonlinearities import rectify, sigmoid

import rechitmodelutils

import numpy as np
import theano.tensor as T
import math

#----------------------------------------------------------------------
# model
#----------------------------------------------------------------------

def makeMultipleConvolutionsAndMaxPool(inputLayer, convSpecs):
    # makes multiple multiple parallel convolutions
    # and adds a maxpool layer to each 

    outputs = []

    totNumOutputElements = 0

    for convSpec in convSpecs:
        convLayer = Conv2DLayer(
            inputLayer,
            num_filters = convSpec['numFilters'],
            filter_size = convSpec['size'],
            nonlinearity = rectify,
            pad = 'same',
            W = GlorotUniform(),
            )

        
        # produce a 1x1 output
        pooling = MaxPool2DLayer(convLayer, 
                                 pool_size = inputLayerDimension,
                                 # pad = ((poolsize - 1) / 2, (poolsize - 1) / 2)
                                 )




        reshaped = ReshapeLayer(pooling,
                                shape = (-1,              # minibatch dimension
                                          convSpec['numFilters']
                                         )
                               )



        totNumOutputElements += convSpec['numFilters']


        outputs.append(reshaped)
        
    # end of loop over convolutional filter specifications
    
    # combine the outputs using a ConcatLayer
    # axis 1 (second dimension) is the layer dimension
    network = ConcatLayer(outputs,
                          axis = 1,
                          )



    return network


#----------------------------------------------------------------------

def makeModel():

    # note that we need several input variables here
    # 3D tensor
    inputVarRecHits        = T.tensor4('rechits')

    # mask for setting rechits to zero
    recHitsInputMask       = T.tensor3('recHitsMask')
    # fill this with zeros and ones

    ninputLayers = 1
    inputLayer = InputLayer(shape=(None, 
                                1,                       # number of input layers
                                inputLayerDimension[0],  # ieta
                                inputLayerDimension[1]   # iphi
                                ),
                         input_var = inputVarRecHits,
                         name = 'rechits',
                        )

    #----------
    # convolutional filters of various sizes
    #----------
    
    # we can't make a maxpool down to 1x1, the maximum is likely
    # to come always from the center of the shower
    # 
    # we would need to look at the k highest values,
    # see https://github.com/Theano/Theano/issues/5608
    #
    # we shadow (zero) the center 5x5 part of the convolutions *output*
    # so the values we look at are at least half of the window
    # away from the center (there is no point in making a copy of the rechits
    # to zero a window in the center because if we all have negative
    # weights in a kernel the largest value is exactly the one
    # in the center)

    #----------

    network = makeMultipleConvolutionsAndMaxPool(inputLayer, filterSpecs['globalFilters'])

    # just feed the output of all the convolutional/maxpool outputs into a dense network for the moment

    numNodes = network.output_shape[1]

    network = DenseLayer(
        network,
        num_units = 2 * numNodes,
        W = GlorotUniform(),
        nonlinearity = rectify)

    network = DenseLayer(
        network,
        num_units = 2 * numNodes,
        W = GlorotUniform(),
        nonlinearity = rectify)

    network = DropoutLayer(network, p = 0.5)

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
