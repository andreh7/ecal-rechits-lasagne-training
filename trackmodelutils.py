#!/usr/bin/env python

import math

from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer, ReshapeLayer, ConcatLayer
from lasagne.init import GlorotUniform
import lasagne.layers

import numpy as np

from lasagne.nonlinearities import rectify, softmax

#----------------------------------------------------------------------

# parameters for track histograms
trkAbsDetaMax = 0.4
trkAbsDphiMax = 0.4

trkDetaBinWidth, trkDphiBinWidth = 2 * math.pi / 360.0, 2 * math.pi / 360.0

# for radial histogram
trkDrMax = 0.4
trkDrBinWidth = 2 * math.pi /360.0

#----------------------------------------------------------------------

# like numpy.arange but including the upper end
def myArange(start, stop, step):
    value = start
    while True:
        yield value
        
        if value >= stop:
            break

        value += step

#----------------------------------------------------------------------

def makeSymmetricBinning(maxVal, step):
    bins = list(myArange(0, maxVal, step))

    # symmetrize
    # but avoid doubling the zero value
    return [ -x for x in bins[::-1][:-1]] + bins

#----------------------------------------------------------------------

# note that we do NOT need makeSymmetricBinning(..) here
# because dr does not go negative
trkBinningDr = list(myArange(0, trkDrMax, trkDrBinWidth))
    
trkBinningDeta = makeSymmetricBinning(trkAbsDetaMax, trkDetaBinWidth)
trkBinningDphi = makeSymmetricBinning(trkAbsDphiMax, trkDphiBinWidth)

#----------------------------------------------------------------------

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

#----------------------------------------------------------------------


class TrackHistograms2d:
    def __init__(self, 
                 trkBinningDeta,
                 trkBinningDphi,
                 relptWeighted,
                 ):
        self.trkBinningDeta = trkBinningDeta
        self.trkBinningDphi = trkBinningDphi
        self.relptWeighted = relptWeighted

    #----------------------------------------

    def make(self, dataset, rowIndices):
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
                           len(self.trkBinningDeta) - 1,
                           len(self.trkBinningDphi) - 1,
                           ), dtype = 'float32')

        for row,rowIndex in enumerate(rowIndices):

            indexOffset = dataset['tracks']['firstIndex'][rowIndex]

            detaValues = []
            dphiValues = []
            if self.relptWeighted:
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

                if self.relptWeighted:
                    weights.append(dataset['tracks']['relpt'][index])

            # end of loop over all tracks of event

            # fill the histogram
            histo, binBoundariesX, binBoundariesY = np.histogram2d(detaValues, 
                                                                   dphiValues,
                                                                   bins = [ self.trkBinningDeta, self.trkBinningDphi],
                                                                   weights = weights)
            retval[row,0,:,:] = histo

        # end of loop over events in this minibatch

        return retval

    #----------------------------------------

#----------------------------------------------------------------------

def makeRadialTracksHistogramModel(input_var):
    # non-convolutional network for the moment
    #
    # note that we produce a model with many outputs,
    # it is up to the calling function to add an output
    # softmax/sigmoid layer or feed this into 
    # another network


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

        # ReLU
        nonlinearity = rectify

        num_units = nodesPerHiddenLayer

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
def make2DTracksHistogramModel(input_var):
    # convolutional network 

    num_filters = [ 64, 64 ]
    filtsize = 5
    poolsize = 2

    # subtract one because the binning arrays include the last upper edge
    width  = len(trkBinningDeta) - 1
    height = len(trkBinningDphi) - 1

    # 2D convolution layers require a dimension for the input channels
    network = InputLayer(shape=(None, 1, width, height),
                         input_var = input_var
                         )

    #----------
    # stage 1 : filter bank -> squashing -> L2 pooling -> normalization
    #----------

    network = Conv2DLayer(
        network, 
        num_filters = num_filters[0], 
        filter_size = (filtsize, filtsize),
        nonlinearity = rectify,
        pad = 'same',
        W = GlorotUniform(),
        )

    network = MaxPool2DLayer(network, pool_size = (poolsize, poolsize),
                             pad = ((poolsize - 1) / 2, (poolsize - 1) / 2)
                             )

    #----------
    # stage 2 : filter bank -> squashing -> L2 pooling -> normalization
    #----------

    network = Conv2DLayer(
        network, 
        num_filters = num_filters[1], 
        filter_size = (3, 3),
        nonlinearity = rectify,
        pad = 'same',
        W = GlorotUniform(),
        )

    network = MaxPool2DLayer(network, pool_size = (poolsize, poolsize),
                             pad = ((poolsize - 1) / 2, (poolsize - 1) / 2)
                             )

    #----------
    # stage 3 : standard 2-layer neural network
    #----------

    thisShape = lasagne.layers.get_output_shape(network,
                                                           (1, 1, width, height))
    print "output shape=", thisShape

    network = ReshapeLayer(network,
                           shape = (-1,              # minibatch dimension
                                     thisShape[1] * thisShape[2] * thisShape[3])
                           )

    # it looks like Lasagne scales the inputs at training time
    # while Torch scales them at inference time ?
    network = DropoutLayer(network, p = 0.5)

    network = DenseLayer(network,
                         num_units = 128,
                         W = GlorotUniform(),
                         nonlinearity = rectify
                         )

    return network

#----------------------------------------------------------------------

