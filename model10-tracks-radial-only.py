#!/usr/bin/env python

# like model06 but with dropout layer only applied
# to the rechits variables, not the other (track iso)
# variables


from lasagne.layers import DenseLayer

from lasagne.init import GlorotUniform
from lasagne.nonlinearities import softmax

import theano.tensor as T
import math

from trackmodelutils import makeTrackHistogramsRadial, makeRadialTracksHistogramModel

#----------------------------------------------------------------------
# model
#----------------------------------------------------------------------

# size of minibatch
batchSize = 32

# how many minibatches to unpack at a time
# and to store in the GPU (to have fewer
# data transfers to the GPU)
batchesPerSuperBatch = math.floor(3345197 / batchSize)

#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

def makeInput(dataset, rowIndices, inputDataIsSparse):
    assert inputDataIsSparse,"non-sparse input data is not supported"

    return [ 
        makeTrackHistogramsRadial(dataset, rowIndices, relptWeighted = True)
        ]


#----------------------------------------------------------------------

def makeModel():

    # note that we need several input variables here
    # 3D tensor
    inputVarTracks         = T.matrix('tracks')

    tracksModel = makeRadialTracksHistogramModel(inputVarTracks)

    # output
    network = DenseLayer(
        tracksModel,
        num_units = 2,  # need two class classification, seems not to work well with sigmoid
        nonlinearity = softmax,
        W = GlorotUniform(),
        )

    return [ inputVarTracks ], network

#----------------------------------------------------------------------
