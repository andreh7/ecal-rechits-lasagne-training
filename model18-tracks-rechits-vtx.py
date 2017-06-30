#!/usr/bin/env python

# like model12-tracks-rechits-vtx.py but with vertex
# index information (such that we can better reproduce
# the selected and worst vertex track isolation 
# as calculated in flashgg)

from lasagne.layers import InputLayer, DenseLayer, ConcatLayer

from lasagne.init import GlorotUniform
from lasagne.nonlinearities import softmax

import theano.tensor as T
import math

import trackmodelutils, rechitmodelutils


#----------------------------------------------------------------------
# (default) model parameters
#----------------------------------------------------------------------

# size of input layer
#
# we use the wide rechits window because
# we also want the wide track window
# (but we could actually restrict the rechits
# to a narrow window and just let the outer
# rechits be zero in the network input without shifting)

inputLayerDimension = (35,35)

isBarrel = True

# size of minibatch
batchSize = 32

#----------------------------------------------------------------------
# hidden units, filter sizes for convolutional network
nstates = [64,64,128]
filtsize = 5
poolsize = 2

#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

# one degree: ~ size of a crystal in the barrel
trackBinWidth = 2 * math.pi / 360.0

# make a binning with fixed bin size and given number of bins
tracksBinningDeta = trackmodelutils.makeBinningFromWidthAndNumber(trackBinWidth, inputLayerDimension[0])
tracksBinningDphi = trackmodelutils.makeBinningFromWidthAndNumber(trackBinWidth, inputLayerDimension[1])

# the binning contains also the upper edge of the last bin
assert len(tracksBinningDeta) == inputLayerDimension[0] + 1
assert len(tracksBinningDphi) == inputLayerDimension[0] + 1

# the center point of the rechits in the dataset is
# at 18,18 in Torch coordinates (one based)
# i.e. at 17,17 in python coordinates (zero based)
# so we have (0..34) x (0..34) in python coordinates
# 
# we make the tracking such that 

unpacker = rechitmodelutils.RecHitsUnpacker(
    width  = inputLayerDimension[0],
    height = inputLayerDimension[1],

    # for shifting 18,18 to 4,12
    # recHitsXoffset = -18 + 4,
    # recHitsYoffset = -18 + 12,
    )

# make a consistent binning for the tracks
trackHistogramMaker = trackmodelutils.TrackHistograms2d(tracksBinningDeta,
                                                        tracksBinningDphi,
                                                        trackWeightFunction = lambda dataset, photonIndex, trackIndex: dataset['tracks']['pt'][trackIndex]
                                                        )
                                        


def makeInput(dataset, rowIndices, inputDataIsSparse):
    assert inputDataIsSparse,"non-sparse input data is not supported"

    # maximum distance (in cm) for which a vertex is considered to be
    # 'the same' or not
    maxVertexDist = 0.01 # 100 um

    detaDphiFunc = lambda dataset, photonIndex, trackIndex: (
        dataset['tracks']['etaAtVertex'][trackIndex] - dataset['phoVars/maxRecHitEta'][photonIndex],
        dataset['tracks']['phiAtVertex'][trackIndex] - dataset['phoVars/maxRecHitPhi'][photonIndex],
        )

    retval = [ 
        # unpack rechits
        unpacker.unpack(dataset, rowIndices),
        ]

    for trackFilter in (
        # tracks from same vertex as diphoton candidate
        lambda dataset, photonIndex, trackIndex: dataset['tracks']['vtxIndex'][trackIndex] == dataset['phoVars/phoVertexIndex'][photonIndex],

        # tracks from the worst iso vertex 
        lambda dataset, photonIndex, trackIndex: dataset['tracks']['vtxIndex'][trackIndex] == dataset['phoVars/phoWorstIsoVertexIndex'][photonIndex],

        # tracks from the second worst iso vertex
        lambda dataset, photonIndex, trackIndex: dataset['tracks']['vtxIndex'][trackIndex] == dataset['phoVars/phoSecondWorstIsoVertexIndex'][photonIndex],

        # tracks from other vertices
        lambda dataset, photonIndex, trackIndex: dataset['tracks']['vtxIndex'][trackIndex] != dataset['phoVars/phoVertexIndex'][photonIndex] and \
                                                 dataset['tracks']['vtxIndex'][trackIndex] != dataset['phoVars/phoWorstIsoVertexIndex'][photonIndex] and \
                                                 dataset['tracks']['vtxIndex'][trackIndex] != dataset['phoVars/phoSecondWorstIsoVertexIndex'][photonIndex],

        ):
        retval.append(
            trackHistogramMaker.make(dataset, rowIndices, 
                                     
                                     detaDphiFunc = detaDphiFunc,
                                     
                                     trackFilter = trackFilter)
            )

    return retval



#----------------------------------------------------------------------

def makeModel():

    inputVars = []
    inputLayers = []

    # make two input variables: one for rechits, one for tracks
    #
    inputVarRecHits        = T.tensor4('rechits')
    inputVars.append(inputVarRecHits)

    # 2D convolution layers require a dimension for the input channels
    inputLayerRecHits = InputLayer(shape=(None, 1 , inputLayerDimension[0], inputLayerDimension[1]),
                                       input_var = inputVarRecHits,
                                       name = 'rechits',
                                       )

    inputLayers.append(inputLayerRecHits)

    for varname in [
        'tracksSameVtx',        # tracks from same vertex as diphoton candidate
        'tracksWorstVtx',       # tracks from the worst iso vertex 
        'tracksSecondWorstVtx', # tracks from the second worst iso vertex
        'tracksOtherVtx',       # tracks from other vertices
        ]:

        # create an input variable
        inputVar = T.tensor4(varname)
        inputVars.append(inputVar)

        # create an input layer associated to this variable
        inputLayer = InputLayer(shape=(None, 1 , inputLayerDimension[0], inputLayerDimension[1]),
                                input_var = inputVar,
                                name = varname,
                                )

        inputLayers.append(inputLayer)

    # end of loop over different vertex types

    # combine them using a ConcatLayer
    # axis 1 (second dimension) is the layer dimension

    network = ConcatLayer(inputLayers,
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

    return inputVars, network

#----------------------------------------------------------------------
