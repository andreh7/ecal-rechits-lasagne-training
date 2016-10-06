#!/usr/bin/env python

# like model06 but with dropout layer only applied
# to the rechits variables, not the other (track iso)
# variables


from lasagne.layers import InputLayer, DenseLayer, ConcatLayer
from lasagne.init import GlorotUniform
from lasagne.nonlinearities import rectify, softmax

import rechitmodelutils

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

def makeModel():

    # note that we need several input variables here
    # 3D tensor
    inputVarRecHits        = T.tensor4('rechits')
    inputVarTrackIsoChosen = T.matrix('trackIsoChosen')
    inputVarTrackIsoWorst  = T.matrix('trackIsoWorst')

    recHitsModel = rechitmodelutils.makeRecHitsModel(inputVarRecHits, width, height, nstates[:2], filtsize, poolsize)

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

unpacker = rechitmodelutils.RecHitsUnpacker(
    width,
    height,
    # for shifting 18,18 to 4,12

    recHitsXoffset = -18 + 4,
    recHitsYoffset = -18 + 12,
    )

def makeInput(dataset, rowIndices, inputDataIsSparse):
    assert inputDataIsSparse,"non-sparse input data is not supported"

    #----------
    # unpack the sparse data
    #----------
    recHits = unpacker.unpack(dataset, rowIndices)

    return [ recHits, 
             dataset['chgIsoWrtChosenVtx'][rowIndices],
             dataset['chgIsoWrtWorstVtx'][rowIndices],
             ]

# ----------------------------------------------------------------------
# function makeInputView(inputValues, first, last)
# 
#   assert(first >= 1)
#   assert(last <= inputValues[1]:size()[1])
# 
#   return { inputValues[1]:sub(first,last),
#            inputValues[2]:sub(first,last),
#            inputValues[3]:sub(first,last)
#            }
#   
# 
# end

# ----------------------------------------------------------------------
