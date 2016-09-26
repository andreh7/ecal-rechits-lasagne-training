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

#----------------------------------------------------------------------
# model
#----------------------------------------------------------------------

# 2-class problem
noutputs = 2

# input dimensions
width = 7
height = 23

ninputs = 1 * width * height

# hidden units, filter sizes for convolutional network
nstates = [64,64,128]
filtsize = 5
poolsize = 2

#----------------------------------------

# size of minibatch
batchSize = 32

# how many minibatches to unpack at a time
# and to store in the GPU (to have fewer
# data transfers to the GPU)
batchesPerSuperBatch = math.floor(3345197 / batchSize)

#----------------------------------------------------------------------

def makeRecHitsModel(input_var):
    # a typical modern convolution network (conv+relu+pool)

    # TODO: check ordering of width and height
    # 2D convolution layers require a dimension for the input channels
    network = InputLayer(shape=(None, 1, width, height),
                         input_var = input_var
                         )


    # see https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialModules
    # stage 1 : filter bank -> squashing -> L2 pooling -> normalization

    ### recHitsModel:add(nn.SpatialConvolutionMM(nfeats,             -- nInputPlane
    ###                                   nstates[1],         -- nOutputPlane
    ###                                   filtsize,           -- kernel width
    ###                                   filtsize,           -- kernel height
    ###                                   1,                  -- horizontal step size
    ###                                   1,                  -- vertical step size
    ###                                   (filtsize - 1) / 2, -- padW
    ###                                   (filtsize - 1) / 2 -- padH
    ###                             ))
    ### recHitsModel:add(nn.ReLU())
    
    network = Conv2DLayer(
        network, 
        num_filters = nstates[0], 
        filter_size = (filtsize, filtsize),
        nonlinearity = rectify,
        pad = 'same',
        W = GlorotUniform(),
        )

    # see https://github.com/torch/nn/blob/master/doc/convolution.md#nn.SpatialMaxPooling
    # recHitsModel:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

    network = MaxPool2DLayer(network, pool_size = (poolsize, poolsize),
                             pad = ((poolsize - 1) / 2, (poolsize - 1) / 2)
                             )

    # stage 2 : filter bank -> squashing -> L2 pooling -> normalization
    ### recHitsModel:add(nn.SpatialConvolutionMM(nstates[1],         -- nInputPlane
    ###                                   nstates[2],         -- nOutputPlane
    ###                                   3,                  -- kernel width
    ###                                   3,                  -- kernel height
    ###                                   1,                  -- horizontal step size
    ###                                   1,                  -- vertical step size
    ###                                   (3 - 1) / 2, -- padW
    ###                                   (3 - 1) / 2 -- padH
    ###                             ))
    ### recHitsModel:add(nn.ReLU())

    network = Conv2DLayer(
        network, 
        num_filters = nstates[1], 
        filter_size = (3, 3),
        nonlinearity = rectify,
        pad = 'same',
        W = GlorotUniform(),
        )

    ### recHitsModel:add(nn.SpatialMaxPooling(poolsize, -- kernel width
    ###                                poolsize, -- kernel height
    ###                                poolsize, -- dW step size in the width (horizontal) dimension 
    ###                                poolsize,  -- dH step size in the height (vertical) dimension
    ###                                (poolsize - 1) / 2, -- pad size
    ###                                (poolsize - 1) / 2 -- pad size
    ###                          ))

    network = MaxPool2DLayer(network, pool_size = (poolsize, poolsize),
                             pad = ((poolsize - 1) / 2, (poolsize - 1) / 2)
                             )

    # stage 3 : standard 2-layer neural network

    # see https://github.com/torch/nn/blob/master/doc/simple.md#nn.View
    # recHitsModel:add(nn.View(nstates[2]*1*5))
    network = ReshapeLayer(network,
                           shape = (-1,              # minibatch dimension
                                     nstates[1]*1*5)
                           )

    # recHitsModel:add(nn.Dropout(0.5))
    # it looks like Lasagne scales the inputs at training time
    # while Torch scales them at inference time ?
    network = DropoutLayer(network, p = 0.5)

    return network

#----------------------------------------------------------------------

def makeModel():

    # note that we need several input variables here
    # 3D tensor
    inputVarRecHits        = T.tensor4('rechits')
    inputVarTrackIsoChosen = T.matrix('trackIsoChosen')
    inputVarTrackIsoWorst  = T.matrix('trackIsoWorst')

    recHitsModel = makeRecHitsModel(inputVarRecHits)

    inputLayerTrackIsoChosen = InputLayer(shape = (None,1), input_var = inputVarTrackIsoChosen)
    inputLayerTrackIsoWorst  = InputLayer(shape = (None,1), input_var = inputVarTrackIsoWorst)

    ### #----------
    ### # track isolation variables 
    ### # ----------
    ### # see e.g. http://stackoverflow.com/questions/32630635/torch-nn-handling-text-and-numeric-input
    ### 
    ### trackIsoModelChosen = nn.Identity()
    ### trackIsoModelWorst  = nn.Identity() 
    ### 
    ### #----------
    ### # put rechits and track iso networks in parallel
    ### #----------
    ### parallelModel = nn.ParallelTable()
    ### parallelModel:add(recHitsModel)
    ### parallelModel:add(trackIsoModelChosen)
    ### parallelModel:add(trackIsoModelWorst)


    #----------
    # combine nn output from convolutional layers for
    # rechits with track isolation variables
    #----------

    network = ConcatLayer([ recHitsModel, inputLayerTrackIsoChosen, inputLayerTrackIsoWorst ],
                          axis = 1)

    #----------
    # common output part
    #----------
    # outputModel:add(nn.Linear(nstates[2]*1*5 + 2, # +2 for the track isolation variables
    #                           nstates[3]))
    # outputModel:add(nn.ReLU())

    network = DenseLayer(
        network,
        num_units = nstates[2],
        W = GlorotUniform(),
        nonlinearity = rectify)

    # output
    network = DenseLayer(
        network,
        num_units = 2,  # need two class classification, seems not to work well with sigmoid
        nonlinearity = softmax,
        W = GlorotUniform(),
        )

    return [ inputVarRecHits, inputVarTrackIsoChosen, inputVarTrackIsoWorst ], network


#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

# parameters of track histogram
trkAbsDetaMax = 0.4
trkAbsDphiMax = 0.4

trkDetaBinWidth, trkDphiBinWidth = 2 * math.pi / 360.0, 2 * math.pi / 360.0

# for radial histogram
trkDrMax = 0.4
trkDrBinWidth = 2 * math.pi /360.0

#----------------------------------------

# like numpy.arange but including the upper end
def myArange(start, stop, step):
    value = start
    while True:
        yield value
        
        if value >= stop:
            break

        value += step

#----------------------------------------

def makeSymmetricBinning(maxVal, step):
    bins = list(myArange(0, maxVal, step))

    # symmetrize
    # but avoid doubling the zero value
    return [ -x for x in bins[::-1][:-1]] + bins

    
#----------------------------------------

# note that we do NOT need makeSymmetricBinning(..) here
# because dr does not go negative
trkBinningDr = list(myArange(0, trkDrMax, trkDrBinWidth))

def makeTrackHistograms(dataset, rowIndices):
    # fills tracks into a histogram
    # for each event
    assert False, "not implemented yet"

#----------------------------------------

def makeTrackHistogramsRadial(dataset, rowIndices, relptWeighted):
    # fills tracks into a histogram
    # for each event

    # note that we need to 'unpack' the tracks
    # and we want a histogram for each entry in rowIndices

    batchSize = len(rowIndices)

    retval = np.empty((batchSize, len(trkBinningDr) - 1))

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

# ----------------------------------------------------------------------
