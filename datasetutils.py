#!/usr/bin/env python

import numpy as np

#----------------------------------------------------------------------

def getActualSize(sizeSpec, loadedData):
    # given a potentially relative size specification ( value between 0 and 1)
    # or None ('take all data') or integer (exact number of items to
    # be taken) returns the number of items to be taken from this data
    
    # use target variable to infer size of loaded dataset
    loadedSize = len(loadedData['y'])

    if sizeSpec == None:
        # None specified, take all loaded data
        return loadedSize

    if sizeSpec < 1:
        assert sizeSpec >= 0
        return int(sizeSpec * loadedSize + 0.5)

    # absolute number of events given (assume this is an integer...)
    return min(sizeSpec, loadedSize)

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

    def normalizeWeights(self):
        # normalizes the weights 
        # to have an average
        # of one per sample
        # 
        # (weights should in principle directly
        # affect the effective learning rate of SGD)

        self.data['weights'] *= (self.data['weights'].shape[0] / float(self.data['weights'].sum()))

#----------------------------------------------------------------------

class SimpleVariableConcatenator:
    # concatenator for 'simple' variables which are just 1D float tensors
    # keeping individual numpy 1D arrays per variable in a dict

    def __init__(self, varnames):
        # note that varnames is treated as sorted
        # so that we get reproducible results
        # (i.e. the order is important when mapping to the input neurons)

        self.data = None
        self.totsize = 0

        # TODO: also support variable names with dots in them indicating
        # that they are part of a lua table
        self.varnames = varnames

        self.data = None

    #----------------------------------------

    def add(self, loaded, thisSize):
        if self.data == None:
            #----------
            # first file 
            #----------

            # fill the individual variables
            self.data = {}
            for varname in self.varnames:
                # store additional variables by name, not by index
                self.data[varname] = loaded[varname].asndarray()[:thisSize].astype('float32').reshape((-1,1))
        else:
            #----------
            # append
            #----------
            # concatenate auxiliary variables
            for varname in self.varnames:
                self.data[varname] = np.concatenate([ self.data[varname], loaded[varname].asndarray()[:thisSize].astype('float32').reshape((-1,1)) ])

    #----------------------------------------

    def normalize(self):
        # normalize each variable individually to zero mean
        # and unit variance 

        # if a variable has zero variance to start with, 
        # do not normalize the variance but this also
        # implies that all values are the same, i.e. the 
        # variable does not contain any information
        for varname in self.varnames:
            self.data[varname] -= self.data[varname].mean()

        print "stddevs before:", [ self.data[varname].std() for varname in self.varnames ]

        for varname in self.varnames:
            stddev = self.data[varname].std()
            if stddev > 0:
                self.data[varname] /= stddev
            else:
                print "WARNING: variable",varname,"has zero standard deviation"

        print "stddevs after:", [ self.data[varname].std() for varname in self.varnames ]



#----------------------------------------------------------------------

class SimpleVariableConcatenatorToMatrix:
    # similar to SimpleVariableConcatenator but producing a 2D 
    # matrix instead of keeping per variable 1D arrays

    #----------------------------------------

    def __init__(self, varnames):
        # note that varnames is treated as sorted
        # so that we get reproducible results
        # (i.e. the order is important when mapping to the input neurons)

        self.data = None
        self.totsize = 0

        # TODO: also support variable names with dots in them indicating
        # that they are part of a lua table
        self.varnames = varnames

        self.data = None

    #----------------------------------------

    def add(self, loaded, thisSize):
        pass
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
                              
