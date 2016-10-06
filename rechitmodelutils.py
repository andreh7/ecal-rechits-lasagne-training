#!/usr/bin/env python

import numpy as np

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
