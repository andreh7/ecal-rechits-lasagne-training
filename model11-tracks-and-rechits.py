#!/usr/bin/env python

# like model06 but with dropout layer only applied
# to the rechits variables, not the other (track iso)
# variables


from lasagne.layers import InputLayer, DenseLayer, ConcatLayer

from lasagne.init import GlorotUniform
from lasagne.nonlinearities import softmax

import theano.tensor as T
import math

import trackmodelutils, rechitmodelutils

#----------------------------------------------------------------------
# hidden units, filter sizes for convolutional network
nstates = [64,64,128]
filtsize = 5
poolsize = 2

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


# input dimensions for rechits
# width, height = 7, 23

# we use the wide rechits window because
# we also want the wide track window
# (but we could actually restrict the rechits
# to a narrow window and just let the outer
# rechits be zero in the network input without shifting)
width, height = 35, 35

# one degree: ~ size of a crystal in the barrel
trackBinWidth = 2 * math.pi / 360.0

# make a binning with fixed bin size and given number of bins
tracksBinningDeta = trackmodelutils.makeBinningFromWidthAndNumber(trackBinWidth, width)
tracksBinningDphi = trackmodelutils.makeBinningFromWidthAndNumber(trackBinWidth, height)

# the binning contains also the upper edge of the last bin
assert len(tracksBinningDeta) == width + 1
assert len(tracksBinningDphi) == height + 1

# the center point of the rechits in the dataset is
# at 18,18 in Torch coordinates (one based)
# i.e. at 17,17 in python coordinates (zero based)
# so we have (0..34) x (0..34) in python coordinates
# 
# we make the tracking such that 

unpacker = rechitmodelutils.RecHitsUnpacker(
    width = width,
    height = height,

    # for shifting 18,18 to 4,12
    # recHitsXoffset = -18 + 4,
    # recHitsYoffset = -18 + 12,
    )

# make a consistent binning for the tracks
trackHistogramMaker = trackmodelutils.TrackHistograms2d(tracksBinningDeta,
                                                        tracksBinningDphi,
                                                        relptWeighted = True)
                                        


def makeInput(dataset, rowIndices, inputDataIsSparse):
    assert inputDataIsSparse,"non-sparse input data is not supported"

    return [ 
        # unpack rechits
        unpacker.unpack(dataset, rowIndices),
        trackHistogramMaker.make(dataset, rowIndices)
        ]


#----------------------------------------------------------------------

def makeModel():

    # make two input variables: one for rechits, one for tracks
    #
    inputVarRecHits        = T.tensor4('rechits')
    inputVarTracks         = T.tensor4('tracks')

    # 2D convolution layers require a dimension for the input channels
    inputLayerRecHits = InputLayer(shape=(None, 1 , width, height),
                                   input_var = inputVarRecHits
                                   )


    inputLayerTracks = InputLayer(shape=(None, 1 , width, height),
                                  input_var = inputVarTracks
                                  )

    # combine them using a ConcatLayer
    # axis 1 (second dimension) is the layer dimension

    network = ConcatLayer([ inputLayerRecHits, inputLayerTracks ],
                          axis = 1,
                          )

    # print "ZZ",dir(network)
    print "output of concat layer:", network.output_shape

    # make a combined model with two layers (one for rechits,
    # one for tracks)

    network = rechitmodelutils.makeRecHitsModel(network, nstates[:2], filtsize, poolsize)

    # output
    network = DenseLayer(
        network,
        num_units = 2,  # need two class classification, seems not to work well with sigmoid
        nonlinearity = softmax,
        W = GlorotUniform(),
        )

    return [ inputVarRecHits, inputVarTracks ], network

#----------------------------------------------------------------------
