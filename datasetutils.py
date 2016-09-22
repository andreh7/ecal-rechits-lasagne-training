#!/usr/bin/env python

import numpy as np

#----------------------------------------------------------------------

def getActualSize(sizeSpec, loadedData):
    # given a potentially relative size specification ( value between 0 and 1)
    # or None ('take all data') or integer (exact number of items to
    # be taken) returns the number of items to be taken from this data
    
    # use target variable to infer size of loaded dataset
    loadedSize = len(loadedData['y'])

    if sizeSpec != None and sizeSpec < 1:
        retval = int(sizeSpec * loadedSize + 0.5)
    else:
        if size != None:
            # absolute number of events given (assume this is an integer...)
            retval = size
        else:
            # None specified, take all loaded data
            retval = loadedSize

        retval = min(thisSize, loadedSize)

    return retval

#----------------------------------------------------------------------

class CommonDataConcatenator:
    # concatenator for commonly used data fields such as the
    # target variable, event weights etc.
    
    #----------------------------------------

    def __init__(self):
        self.data = None
        self.totsize = 0

        self.data = None

    #----------------------------------------

    def add(self, loaded, thisSize):
        # @param loaded is the data to be added
        self.totsize += thisSize
        
        if self.data == None:
    
            #----------
            # create the first entry
            #----------

            self.data = dict(
               data    = {},
            
               # labels are 0/1 because we use cross-entropy loss
               labels  = loaded['y'].asndarray()[:thisSize].astype('float32'),
            
               weights = loaded['weight'].asndarray()[:thisSize].astype('float32'),
          
               mvaid   = loaded['mvaid'].asndarray()[:thisSize].astype('float32'),
            )

      
        else:

            #----------
            # append
            #----------          

            self.data['labels']  = np.concatenate((self.data['labels'],  loaded['y'].asndarray()[:thisSize].astype('float32')))
            self.data['weights'] = np.concatenate((self.data['weights'], loaded['weight'].asndarray()[:thisSize].astype('float32')))
            self.data['mvaid']   = np.concatenate((self.data['mvaid'],   loaded['mvaid'].asndarray()[:thisSize].astype('float32')))
            
        # end of appending
        

    #----------------------------------------



#----------------------------------------------------------------------

class SparseConcatenator:

    #----------------------------------------

    def __init__(self, groupVarName, firstIndexVar, numItemsVar, 
                 otherVars
                 ):
        self.data = None
        self.totsize = 0
        
        # e.g. 'X' for rechits
        self.groupVarName = groupVarName
        
        # e.g. 'firstIndex' for rechits
        self.firstIndexVar = firstIndexVar

        # e.g. 'numRecHits' for rechits
        self.numItemsVar = numItemsVar

        # e.g. [ 'x', 'y', 'energy'] for rechits
        assert len(otherVars) > 0, "must specify at least one item in 'otherVars'"
        self.otherVars = otherVars

    #----------------------------------------

    def add(self, loaded, thisSize):
        # @param loaded is the data to be added
        self.totsize += thisSize

        # determine last object index
        if thisSize < loaded[self.groupVarName][self.firstIndexVar].size[0]:
            # the index of the item behind the last one we 
            # should take
            # note that these are one-based, so we substract one
            objectsEndIndex = loaded[self.groupVarName][self.firstIndexVar][thisSize] - 1
        else:
            # we take the entire dataset
            objectsEndIndex = loaded[self.groupVarName][self.otherVars[0]].size[0]

        if self.data == None:
            #----------
            # create the first entry
            #----------
            self.data = dict()
  
            # copy objects (e.g. rechits, tracks)

            # copy the indices and lengths
            # note that the torch firstIndex values are one based, we subtract one here
            self.data[self.firstIndexVar] = loaded[self.groupVarName][self.firstIndexVar].asndarray()[:thisSize] - 1
            self.data[self.numItemsVar] = loaded[self.groupVarName][self.numItemsVar].asndarray()[:thisSize]

            # copy the sparsified data
            for varname in self.otherVars:
                self.data[varname]      = loaded[self.groupVarName][varname].asndarray()[:objectsEndIndex]

        else:

            #----------
            # append
            #----------

            numPhotonsBefore = self.data[self.firstIndexVar].size
            numObjectsBefore = self.data[self.otherVars[0]].size

            # append sparse objects (e.g. rechits, tracks)

            # copy the sparsified information
            for varname in self.otherVars:
                self.data[varname]      = np.concatenate([ self.data[varname],      loaded[self.groupVarName][varname].asndarray()[:objectsEndIndex] ])
      
            # copy the indices and lengths
            # we subtract one here from firstIndex
            self.data[self.firstIndexVar] = np.concatenate([ self.data[self.firstIndexVar], loaded[self.groupVarName][self.firstIndexVar].asndarray()[:thisSize] - 1 ])
            self.data[self.numItemsVar] = np.concatenate([ self.data[self.numItemsVar], loaded[self.groupVarName][self.numItemsVar].asndarray()[:thisSize] ])

            # we have to shift the first indices, they are only valid within a single file
            self.data[self.firstIndexVar][numPhotonsBefore:numPhotonsBefore + thisSize] += numObjectsBefore

#----------------------------------------------------------------------

def makeRecHitsConcatenator():
    return SparseConcatenator("X", 
                              "firstIndex",
                              "numRecHits",
                              ['x', 'y', 'energy'])


#----------------------------------------------------------------------
def makeTracksConcatenator():
    return SparseConcatenator("X", 
                              "firstIndex",
                              "numTracks",
                              ['relpt', 'charge', 'dphiAtVertex', 'detaAtVertex'])


#----------------------------------------------------------------------
                              
