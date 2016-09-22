#!/usr/bin/env python

import numpy as np

#----------------------------------------------------------------------

class RecHitsConcatenator:

    #----------------------------------------

    def __init__(self):
        self.data = None
        self.totsize = 0

    #----------------------------------------

    def add(self, loaded, thisSize):
        # @param loaded is the data to be added
        self.totsize += thisSize

        # determine last rechit index
        if thisSize < loaded['X']['firstIndex'].size[0]:
            # the index of the rechit behind the last one we 
            # should take
            # note that these are one-based, so we substract one
            recHitEndIndex = loaded['X']['firstIndex'][thisSize] - 1
        else:
            # we take the entire dataset
            recHitEndIndex = loaded['X']['energy'].size[0]

        if self.data == None:
            #----------
            # create the first entry
            #----------
            self.data = dict()
  
            # copy rechits

            # copy the indices and lengths
            # note that the torch firstIndex values are one based, we subtract one here
            self.data['firstIndex'] = loaded['X']['firstIndex'].asndarray()[:thisSize] - 1
            self.data['numRecHits'] = loaded['X']['numRecHits'].asndarray()[:thisSize]

            # copy the rechits
            self.data['x']      = loaded['X']['x'].asndarray()[:recHitEndIndex]
            self.data['y']      = loaded['X']['y'].asndarray()[:recHitEndIndex]
            self.data['energy'] = loaded['X']['energy'].asndarray()[:recHitEndIndex]
        else:

            #----------
            # append
            #----------

            numPhotonsBefore = self.data['firstIndex'].size
            numRecHitsBefore = self.data['energy'].size

            # append sparse rechits

            # copy the rechits
            self.data['x']      = np.concatenate([ self.data['x'],      loaded['X']['x'].asndarray()[:recHitEndIndex] ])
            self.data['y']      = np.concatenate([ self.data['y'],      loaded['X']['y'].asndarray()[:recHitEndIndex] ])
            self.data['energy'] = np.concatenate([ self.data['energy'], loaded['X']['energy'].asndarray()[:recHitEndIndex] ])
      
            # copy the indices and lengths
            # we subtract one here from firstIndex
            self.data['firstIndex'] = np.concatenate([ self.data['firstIndex'], loaded['X']['firstIndex'].asndarray()[:thisSize] - 1 ])
            self.data['numRecHits'] = np.concatenate([ self.data['numRecHits'], loaded['X']['numRecHits'].asndarray()[:thisSize] ])

            # we have to shift the first indices, they are only valid within a single file
            self.data['firstIndex'][numPhotonsBefore:numPhotonsBefore + thisSize] += numRecHitsBefore

#----------------------------------------------------------------------
