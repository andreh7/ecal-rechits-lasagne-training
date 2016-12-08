#!/usr/bin/env python

# model for regression training of track isolation variables
# 



from lasagne.layers import InputLayer, DenseLayer, ConcatLayer

from lasagne.init import GlorotUniform
from lasagne.nonlinearities import rectify, softmax

import theano.tensor as T
import math

import trackmodelutils, rechitmodelutils

#----------------------------------------------------------------------
# hidden units, filter sizes for convolutional network
numHiddenUnits = 128

numHiddenLayers = 1


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
        dataset['sortedTracks'][rowIndices,:],
        dataset['numTracks'][rowIndices],
        ]

#----------------------------------------------------------------------

def makeTarget(dataset, rowIndices):
    # returns the target to be trained

    return dataset['otherVars']['chgIsoWrtChosenVtx'][rowIndices,:].astype('float32')

#----------------------------------------------------------------------

def makeModel():
    # make two input variables: one for 'global' variables such as number of tracks, one for tracks
    #
    inputVarTracks         = T.matrix('trackVars')
    inputVarNumTracks      = T.matrix('numTracks')

    inputVarsTracks = InputLayer(shape=(None, tracksVarMaker.numVars),
                                 input_var = inputVarTracks,
                                 name = 'trackVars',
                                 )

    inputAuxVars = InputLayer(shape=(None, 1),
                                 input_var = inputVarNumTracks,
                                 name = 'numTracks',
                                 )

    #----------
    # combine nn output from convolutional layers for
    # rechits with track isolation variables
    #----------

    network = ConcatLayer([ inputVarsTracks, inputAuxVars ],
                          axis = 1)

    for i in range(numHiddenLayers):

        network = DenseLayer(
            network,
            num_units = numHiddenUnits,
            W = GlorotUniform(),
            nonlinearity = rectify)
        
    #----------
    # output
    #----------
    network = DenseLayer(
        network,
        num_units = 1, 
        nonlinearity = None, # linear output
        W = GlorotUniform(),
        )

    return [ inputVarTracks, inputVarNumTracks ], network

#----------------------------------------------------------------------
