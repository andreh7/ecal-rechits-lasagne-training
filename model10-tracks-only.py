#!/usr/bin/env python

# like model06 but with dropout layer only applied
# to the rechits variables, not the other (track iso)
# variables


from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer, ReshapeLayer, ConcatLayer
from lasagne.init import GlorotUniform
from lasagne.nonlinearities import rectify, softmax

import numpy as np
import theano.tensor as T
import math

from trackmodelutils import myArange, makeSymmetricBinning, trkBinningDr, trkBinningDeta, trkBinningDphi

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

def makeTrackHistograms2D(dataset, rowIndices, relptWeighted):
    # fills tracks into a histogram
    # for each event

    # note that we need to 'unpack' the tracks
    # and we want a histogram for each entry in rowIndices

    batchSize = len(rowIndices)
    
    # first index:  event index (for minibatch)
    # second index: fixed to 1 (convolutional filters seem to need this)
    # third index:  width / deta
    # fourth index: height / dphi
    retval = np.empty((batchSize, 
                       1,
                       len(trkBinningDeta) - 1,
                       len(trkBinningDphi) - 1,
                       ), dtype = 'float32')

    for row,rowIndex in enumerate(rowIndices):

        indexOffset = dataset['tracks']['firstIndex'][rowIndex]

        detaValues = []
        dphiValues = []
        if relptWeighted:
            weights = []
        else:
            weights = None

        #----------
        # unpack the sparse data
        #----------
        for trackIndex in range(dataset['tracks']['numTracks'][rowIndex]):
    
            index = indexOffset + trackIndex

            deta = dataset['tracks']['detaAtVertex'][index]
            dphi = dataset['tracks']['dphiAtVertex'][index]

            detaValues.append(deta)
            dphiValues.append(dphi)

            if relptWeighted:
                weights.append(dataset['tracks']['relpt'][index])

        # end of loop over all tracks of event

        # fill the histogram
        histo, binBoundariesX, binBoundariesY = np.histogram2d(detaValues, 
                                                               dphiValues,
                                                               bins = [ trkBinningDeta, trkBinningDphi],
                                                               weights = weights)
        retval[row,0,:,:] = histo

    # end of loop over events in this minibatch

    return retval

#----------------------------------------

def makeTrackHistogramsRadial(dataset, rowIndices, relptWeighted):
    # fills tracks into a histogram
    # for each event

    # note that we need to 'unpack' the tracks
    # and we want a histogram for each entry in rowIndices

    batchSize = len(rowIndices)

    retval = np.empty((batchSize, len(trkBinningDr) - 1), dtype = 'float32')

    for row,rowIndex in enumerate(rowIndices):

        indexOffset = dataset['tracks']['firstIndex'][rowIndex]

        drValues = []
        if relptWeighted:
            weights = []
        else:
            weights = None

        #----------
        # unpack the sparse data
        #----------
        for trackIndex in range(dataset['tracks']['numTracks'][rowIndex]):
    
            index = indexOffset + trackIndex

            deta = dataset['tracks']['detaAtVertex'][index]
            dphi = dataset['tracks']['dphiAtVertex'][index]

            dr = math.sqrt(deta * deta + dphi * dphi)
            drValues.append(dr)

            if relptWeighted:
                weights.append(dataset['tracks']['relpt'][index])

        # end of loop over all tracks of event

        # fill the histogram
        retval[row,:], binBoundaries = np.histogram(drValues, 
                                                    bins = trkBinningDr,
                                                    weights = weights)
        
    # end of loop over events in this minibatch

    return retval

#----------------------------------------


def makeInput(dataset, rowIndices, inputDataIsSparse):
    assert inputDataIsSparse,"non-sparse input data is not supported"

    return [ 
        makeTrackHistogramsRadial(dataset, rowIndices, relptWeighted = True)
        ]

#----------------------------------------------------------------------

def makeRadialTracksHistogramModel(input_var):
    # non-convolutional network for the moment
    #


    # subtract one because the upper boundary of the last
    # bin is also included
    width = len(trkBinningDr) - 1

    network = InputLayer(shape=(None, width),
                         input_var = input_var
                         )

    # note that the nonlinearities of the Dense
    # layer is applied on the OUTPUT side

    numHiddenLayers = 3
    nodesPerHiddenLayer = width * 2

    for i in range(numHiddenLayers):

        if i < numHiddenLayers - 1:
            # ReLU
            nonlinearity = rectify

            num_units = nodesPerHiddenLayer
        else:
            # add a dropout layer at the end ?

            # sigmoid at output: can't get this
            # to work with minibatch size > 1
            # nonlinearity = sigmoid
            nonlinearity = softmax

            num_units = 2

        network = DenseLayer(network,
                             num_units = num_units,
                             W = GlorotUniform(),
                             nonlinearity = nonlinearity
                             )

    # end of loop over hidden layers



    # it looks like Lasagne scales the inputs at training time
    # while Torch scales them at inference time ?
    # network = DropoutLayer(network, p = 0.5)

    return network

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
