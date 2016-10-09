#!/usr/bin/env python

import numpy as np

from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer, ReshapeLayer
from lasagne.nonlinearities import rectify
from lasagne.init import GlorotUniform


#----------------------------------------------------------------------

class RecHitsUnpacker:
    # fills sparse rechits into a tensor

    #----------------------------------------

    def __init__(self, width, height, recHitsXoffset = 0, recHitsYoffset = 0):
        self.width = width
        self.height = height
        self.recHitsXoffset = recHitsXoffset
        self.recHitsYoffset = recHitsYoffset

    #----------------------------------------

    def unpack(self, dataset, rowIndices):
        batchSize = len(rowIndices)

        recHits = np.zeros((batchSize, 1, self.width, self.height), dtype = 'float32')

        for i in range(batchSize):

            rowIndex = rowIndices[i]

            # we do NOT subtract one because from 'firstIndex' because
            # these have been already converted in the class SparseConcatenator
            # in datasetutils.py
            indexOffset = dataset['rechits']['firstIndex'][rowIndex]

            for recHitIndex in range(dataset['rechits']['numRecHits'][rowIndex]):

                # we subtract -1 from the coordinates because these are one based
                # coordinates for Torch (and SparseConcatenator does NOT subtract this)
                xx = dataset['rechits']['x'][indexOffset + recHitIndex] - 1 + self.recHitsXoffset
                yy = dataset['rechits']['y'][indexOffset + recHitIndex] - 1 + self.recHitsYoffset

                if xx >= 0 and xx < self.width and yy >= 0 and yy < self.height:
                    recHits[i, 0, xx, yy] = dataset['rechits']['energy'][indexOffset + recHitIndex]

            # end of loop over rechits of this photon
        # end of loop over minibatch indices
        
        return recHits

#----------------------------------------------------------------------

def makeRecHitsModel(input_var, width, height, nstates, filtsize, poolsize):
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
    lastMaxPoolOutputShape = network.output_shape
    print "last maxpool layer output:", lastMaxPoolOutputShape

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

