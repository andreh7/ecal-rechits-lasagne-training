#!/usr/bin/env python

import math

import numpy as np

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
def makeBinningFromWidthAndNumber(binWidth, numBins):
    # produces a binning symmetric around zero
    # given a number of total bins and a bin width
    #
    # note that this returns numBins + 1 values because
    # also the upper end of the histogram is returned
    
    if numBins & 1 == 1:
        # odd number of bins
        # there is a 'center' bin crossing zero

        # rounded down
        halfNumBins = (numBins-1) / 2

        bins = np.arange(-halfNumBins - 1, halfNumBins + 1) + 0.5

        assert len(bins) == numBins + 1
        return bins * binWidth
    else:
        # even number of bins
        raise Exception("even number of bins case not yet implemented")

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
                 trackWeightFunction,
                 ):
        # @param trackWeightFunction must be a function taking parameters
        #  (dataset, photonIndex, trackIndex) and return a weight to be
        #  used in the histogram. If trackWeightFunction is None,
        #  weights of all tracks are effectively set to one.

        self.trkBinningDeta = trkBinningDeta
        self.trkBinningDphi = trkBinningDphi
        self.trackWeightFunction = trackWeightFunction

    #----------------------------------------

    def make(self, dataset, rowIndices, detaDphiFunc, trackFilter = None):
        # fills tracks into a histogram
        # for each event

        # note that we need to 'unpack' the tracks
        # and we want a histogram for each entry in rowIndices

        # detaDphiFunc is a function taking (dataset, photonIndex, trackIndex)
        # as an argument and must return a tuple (deta, dphi) of the
        # track w.r.t the photon

        # trackFilter must be a function taking (dataset, photonIndex, trackIndex) as arguments
        # and return True if a track should be added to this histogram

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
            if self.trackWeightFunction != None:
                weights = []
            else:
                weights = None

            #----------
            # unpack the sparse data
            #----------
            for trackIndex in range(dataset['tracks']['numTracks'][rowIndex]):

                index = indexOffset + trackIndex

                #----------
                # apply track filter if given
                #----------
                if trackFilter != None:
                    if not trackFilter(dataset, rowIndex, index):
                        continue

                #----------

                deta, dphi = detaDphiFunc(dataset, rowIndex, index)

                detaValues.append(deta)
                dphiValues.append(dphi)

                if self.trackWeightFunction != None:
                    weights.append(self.trackWeightFunction(dataset, rowIndex, index))

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

    from lasagne.layers import InputLayer, DenseLayer
    from lasagne.init import GlorotUniform

    network = InputLayer(shape=(None, width),
                         input_var = input_var
                         )

    # note that the nonlinearities of the Dense
    # layer is applied on the OUTPUT side

    numHiddenLayers = 3
    nodesPerHiddenLayer = width * 2

    from lasagne.nonlinearities import rectify

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

    from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer, ReshapeLayer
    from lasagne.init import GlorotUniform

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

    import lasagne.layers
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

class TrackVarsMaker:
    # returns variables of the n highest pt tracks

    def __init__(self, 
                 numTracks
                 ):

        # number of tracks to keep for each event
        self.numTracks = numTracks

        # functions which return the value to be
        # stored, given the dataset and an entry index
        # 
        # note that these are called for each of the highest pt rel tracks
        self.extractionFunctions = []
        self.varnamePrefixes = []

        # rel pt of tracks
        self.extractionFunctions.append(lambda dataset, index: dataset['tracks']['relpt'][index])
        self.varnamePrefixes.append("relpt")

        # dz of track vertex
        self.extractionFunctions.append(lambda dataset, index: dataset['tracks']['vtxDz'][index])
        self.varnamePrefixes.append("vtxDz")

        # charge
        self.extractionFunctions.append(lambda dataset, index: dataset['tracks']['charge'][index])
        self.varnamePrefixes.append("charge")

        # deta/dphi of track
        self.extractionFunctions.append(lambda dataset, index: dataset['tracks']['detaAtVertex'][index])
        self.extractionFunctions.append(lambda dataset, index: dataset['tracks']['dphiAtVertex'][index])
        self.varnamePrefixes.append("detaAtVertex")
        self.varnamePrefixes.append("dphiAtVertex")

        #----------
        assert len(self.varnamePrefixes) == len(self.extractionFunctions)

        # expand variable names
        self.varnames = []
        for i in range(self.numTracks):
            for prefix in self.varnamePrefixes:
                self.varnames.append(prefix + "%02d" % i)
        
        self.numVars = self.numTracks * len(self.extractionFunctions)
        assert self.numVars == len(self.varnames)

    #----------------------------------------

    def makeVars(self, dataset, normalizeVars = [], trackSelFunc = None):
        # fills track variables for each event
        # 
        # @param rowIndices is the indices of the rows (events)
        #
        # @param normalizeVars fnmatch expressions of variables to be normalized
        # 
        # @param trackSelFunc a function taking dataset['tracks'] and a track index
        #        which must return True iff the track should be selected.
        #        If None, no selection is applied

        # take the entire dataset
        rowIndices = range(len(dataset['tracks']['numTracks']))

        # note that we need to 'unpack' the tracks
        batchSize = len(rowIndices)

        # first index:  event index (for minibatch)
        # second index: variable index
        retval = np.ones((batchSize, 
                           self.numVars
                           ), dtype = 'float32') * -99.

        varIndex = 0

        # loop over events
        for eventIndex,rowIndex in enumerate(rowIndices):

            indexOffset = dataset['tracks']['firstIndex'][rowIndex]

            # get indices in track array for this event
            trackIndices = [ indexOffset + trackIndex 
                             for trackIndex in range(dataset['tracks']['numTracks'][rowIndex]) ]

            # sort track indices (within an event) by decreasing relpt
            trackIndices.sort(key = lambda index: dataset['tracks']['relpt'][index], 
                              reverse = True)

            #----------
            # unpack the sparse data
            #----------
            varIndex = 0
            storeIndex = 0   # index of the next selected track

            # find which tracks are selected
            if trackSelFunc != None:
                selectedTrackIndices = []

                for trackIndex in trackIndices:

                    # check if track is selected
                    if trackSelFunc(dataset['tracks'], trackIndex):
                        # track is selected
                        selectedTrackIndices.append(trackIndex)
            else:
                selectedTrackIndices = trackIndices

            # loop over all tracks
            for ind, trackIndex in enumerate(selectedTrackIndices):

                # trackIndex is the pointer into the array of the input data

                if ind >= self.numTracks:
                    # more tracks than we want to keep
                    break

                # extract track information for each variable
                for varfunc in self.extractionFunctions: 
                    retval[eventIndex, varIndex] = varfunc(dataset, trackIndex)
                    varIndex += 1

            # end of loop over all tracks of event

        # end of loop over events in this minibatch

        #----------
        # normalize variables if specified
        #----------

        for varIndex, varname in enumerate(self.varnames):
            # check if any of the specified patterns
            # matches
            import fnmatch
            matches = False
            for pattern in normalizeVars:
                if fnmatch.fnmatch(varname, pattern):
                    matches = True
                    break

            if matches:
                print "normalizing track variable",varname,"range before minimization: %f..%f" % (min(retval[:, varIndex]), max(retval[:, varIndex]))
                retval[:,varIndex] -= np.mean(retval[:,varIndex])
                stddev = retval[:,varIndex].std()
                if stddev > 0:
                    retval[:,varIndex] /= stddev
                else:
                    print "WARNING: variable",varname,"has zero standard deviation"

        return retval

    #----------------------------------------
