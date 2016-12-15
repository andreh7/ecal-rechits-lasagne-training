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

    #----------------------------------------

    def normalizeSignalToBackgroundWeights(self):
        # normalize sum of signal weights to be equal to sum of background weights
        sumSig = self.data['weights'][self.data['labels'] == 1].sum()
        sumBg = self.data['weights'][self.data['labels'] != 1].sum()

        self.data['weights'][self.data['labels'] == 1] *= sumBg / float(sumSig)

    #----------------------------------------

    def getNumEntries(self):
        # returns the number of entries (samples) 
        return len(self.data['labels'])

    #----------------------------------------

#----------------------------------------------------------------------

class SimpleVariableConcatenator:
    # concatenator for 'simple' variables which are just 1D float tensors
    # keeping individual numpy 1D arrays per variable in a dict

    def __init__(self, varnames, accessorFuncs = None):
        # note that varnames is treated as sorted
        # so that we get reproducible results
        # (i.e. the order is important when mapping to the input neurons)
        # 
        # @param accessorFuncs: if not None, should be a dict mapping from
        #                       the variable name to a method returning the data given the input vector

        self.data = None
        self.totsize = 0

        # TODO: also support variable names with dots in them indicating
        # that they are part of a lua table
        self.varnames = varnames

        self.data = None
        self.accessorFuncs = accessorFuncs

    #----------------------------------------

    def __getVar(self, loaded, varname):
        if self.accessorFuncs == None or not self.accessorFuncs.has_key(varname):
            # plain access
            return loaded[varname]

        else:
            func = self.accessorFuncs[varname]

            return func(loaded)

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
                loadedVar = self.__getVar(loaded, varname)

                self.data[varname] = loadedVar.asndarray()[:thisSize].astype('float32').reshape((-1,1))
        else:
            #----------
            # append
            #----------
            # concatenate auxiliary variables
            for varname in self.varnames:
                loadedVar = self.__getVar(loaded, varname)
                self.data[varname] = np.concatenate([ self.data[varname], loadedVar.asndarray()[:thisSize].astype('float32').reshape((-1,1)) ])

    #----------------------------------------

    def normalize(self, selectedVars = None, excludedVars = None):
        # normalize each variable individually to zero mean
        # and unit variance 
        # 
        # these are interpreted as fnmatch patterns

        import fnmatch

        #----------

        def anyMatch(patterns, varname):
            # returns True if any of the given patterns
            # matches varname
            for pattern in patterns:
                if fnmatch.fnmatch(varname, pattern):
                    return True

            return False
        #----------
            

        if selectedVars != None and excludedVars != None:
            raise Exception("can't specify selectedVars and excludedVars at the same time")

        if selectedVars == None and excludedVars == None:
            # take all variables
            selectedVars = set(self.varnames)
        elif excludedVars != None:

            # expand patterns
            selectedVars = [ varname for varname in self.varnames if not anyMatch(excludedVars, varname) ]
        else:
            # only selectedVars is not None, expand patterns
            selectedVars = [ varname for varname in self.varnames if anyMatch(excludedVars, varname) ]            

        # keep order
        selectedVars = [ varname for varname in self.varnames if varname in selectedVars ]

        # if a variable has zero variance to start with, 
        # do not normalize the variance but this also
        # implies that all values are the same, i.e. the 
        # variable does not contain any information
        for varname in selectedVars:
            self.data[varname] -= self.data[varname].mean()

        print "stddevs before:", [ self.data[varname].std() for varname in selectedVars ]

        for varname in selectedVars:
            stddev = self.data[varname].std()
            if stddev > 0:
                self.data[varname] /= stddev
            else:
                print "WARNING: variable",varname,"has zero standard deviation"

        print "stddevs after:", [ self.data[varname].std() for varname in selectedVars ]

#----------------------------------------------------------------------

class SimpleVariableConcatenatorToMatrix:
    # similar to SimpleVariableConcatenator but producing a 2D 
    # matrix instead of keeping per variable 1D arrays

    #----------------------------------------

    def __init__(self, groupVarName, varnames):
        # note that varnames is treated as sorted
        # so that we get reproducible results
        # (i.e. the order is important when mapping to the input neurons)

        self.data = None
        self.totsize = 0

        self.groupVarName = groupVarName

        # TODO: also support variable names with dots in them indicating
        # that they are part of a lua table
        self.varnames = varnames

        self.numvars = len(self.varnames)

        self.data = None

    #----------------------------------------

    def add(self, loaded, thisSize):
        if self.data is None:
            #----------
            # first file
            #----------

            # allocate a 2D Tensor
            self.data = np.ndarray((thisSize, self.numvars), dtype = 'float32')
      
            # copy over the individual variables: use a 2D tensor
            # with each column representing a variables
            for varindex, varname in enumerate(self.varnames):
                self.data[:, varindex] = loaded[self.groupVarName][varname].asndarray()[:thisSize]

        else:
            #----------
            # append
            #----------          
            
            # special treatment for input variables
      
            # note that we can not use resize(..) here as the contents
            # of the resized tensor are undefined according to 
            # https://github.com/torch/torch7/blob/master/doc/tensor.md#resizing
            #
            # so we build first a tensor with the new values
            # and then concatenate this to the previously loaded data
            newData = np.ndarray((thisSize, self.numvars), dtype = 'float32')
      
            for varindex, varname in enumerate(self.varnames):
                newData[:,varindex] = loaded['phoIdInput'][varname].asndarray()[:thisSize]
      
            # and append
            self.data = np.concatenate((self.data, newData))

    #----------------------------------------

    def normalize(self):
        # normalize each variable individually to zero mean
        # and unit variance 
        # 
        # if a variable has zero variance to start with, 
        # do not normalize the variance but this also
        # implies that all values are the same, i.e. the 
        # variable does not contain any information

        self.data -= self.data.mean(axis = 0)

        print "stddevs before:",self.data.std(axis = 0)

        for varnum in range(self.numvars):
            std = self.data[:,varnum].std()
            if std > 0:
                self.data[:,varnum] /= std

        print "stddevs after:",self.data.std(axis = 0)

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
def makeTracksConcatenator(additionalVariables = []):
    return SparseConcatenator("tracks", 
                              "firstIndex",
                              "numTracks",
                              ['relpt', 'charge', 'dphiAtVertex', 'detaAtVertex'] + additionalVariables)


#----------------------------------------------------------------------

# 2D reweighting in pt and eta
# standard BDT id reweights signal to have the same weight
# as background

class PtEtaReweighter:

    def __fillHistogram(self, pt, eta):

        print "bins=",[ np.linspace(self.ptBinning['xmin'],  self.ptBinning['xmax'],  self.ptBinning['nbins'] + 1),
                                                            np.linspace(self.etaBinning['xmin'], self.etaBinning['xmax'], self.etaBinning['nbins'] + 1)
                                                            ]
        print "pt=",pt
        print "eta=",eta

        counts, ptEdges, etaEdges = np.histogram2d(pt, eta, 
                                                   bins = [ np.linspace(self.ptBinning['xmin'],  self.ptBinning['xmax'],  self.ptBinning['nbins'] + 1),
                                                            np.linspace(self.etaBinning['xmin'], self.etaBinning['xmax'], self.etaBinning['nbins'] + 1)
                                                            ])
        
        return counts


    def __init__(self, pt, eta, isSignal, isBarrel):

        # 4 GeV bins from 0 to 120
        self.ptBinning = dict(nbins = 30, xmin = 0, xmax = 120)

        if isBarrel:
            self.etaBinning = dict(nbins = 32, xmin = 0, xmax = 1.6)
        else:
            self.etaBinning = dict(nbins = 20, xmin = 1.5, xmax = 2.5)
        
        eta = np.abs(eta)

        #----------
        # separate signal and background
        #----------
        signalPt  = pt[isSignal == 1]
        signalEta = eta[isSignal == 1]

        backgroundPt  = pt[isSignal == 0]
        backgroundEta = eta[isSignal == 0]

        #----------

        # build histograms
        
        self.sigHistogram = self.__fillHistogram(signalPt, signalEta)
        self.bgHistogram  = self.__fillHistogram(backgroundPt, backgroundEta)

        # calculate ratio histogram to reweight signal to background
        self.ratioHistogram = self.bgHistogram / self.sigHistogram

        # do not reweight events where the background is zero
        # otherwise, if we get NaNs, this will in the end
        # make all weights NaNs if we normalize the average
        # weight to one
        self.ratioHistogram[self.bgHistogram == 0] = 1.

        # avoid INF values (leading to NaNs later on)
        # (just leave the corresponding signal weights 
        # unchanged)
        self.ratioHistogram[np.isinf(self.ratioHistogram)] = 1.

        # find NaNs in the ratio histogram: even a single
        # NaN will spoil all event weights because
        # the sum of weights (which is then used
        # to normalize the event weights) is then NaN
        
        if np.isnan(self.ratioHistogram).any():
            raise Exception("2D pt/eta reweighting has NaN values")

        if np.isinf(self.ratioHistogram).any():
            raise Exception("2D pt/eta reweighting has INF values")

    #----------------------------------------


    def __calculateBinIndices(self, binning, values):
        
        binWidth = (binning['xmax'] - binning['xmin']) / binning['nbins']

        binIndices = (values - binning['xmin']) / binWidth

        # round downwards
        binIndices = binIndices.astype('int32')
        
        binIndices = binIndices.clip(0, binning['nbins'] - 1)

        return binIndices

    #----------------------------------------

    def getSignalScaleFactors(self, ptValues, etaValues, isSignal):
        # returns 1 for background entries

        # calculate bin indices
        # (ignore rounding errors)

        etaValues = np.abs(etaValues)

        ptBins = self.__calculateBinIndices(self.ptBinning, ptValues)
        etaBins = self.__calculateBinIndices(self.etaBinning, etaValues)

        factors = self.ratioHistogram[ptBins, etaBins]

        # do not reweight background
        factors[isSignal != 1] = 1.

        return factors



#----------------------------------------------------------------------
                              
