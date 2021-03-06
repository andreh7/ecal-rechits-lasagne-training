# datasets including chose and worst track isolation (2016-05-28)

# same as dataset04-barrel.py but with newer dataset allowing for pt,eta reweighting
datasetDir = '../data/2016-11-18-photon-et'

isBarrel = True

dataDesc = dict(
    train_files = [ datasetDir + '/GJet20to40_rechits-barrel-train.t7',
                    datasetDir + '/GJet20toInf_rechits-barrel-train.t7',
                    datasetDir + '/GJet40toInf_rechits-barrel-train.t7',
                    ],

    test_files  = [ datasetDir + '/GJet20to40_rechits-barrel-test.t7',
                    datasetDir + '/GJet20toInf_rechits-barrel-test.t7',
                    datasetDir + '/GJet40toInf_rechits-barrel-test.t7'
                    ],

    inputDataIsSparse = True,

    # if one specifies nothing (or nil), the full sizes
    # from the input samples are taken
    # 
    # if one specifies values < 1 these are interpreted
    # as fractions of the sample
    # trsize, tesize = 10000, 1000
    # trsize, tesize = 0.1, 0.1
    # trsize, tesize = 0.01, 0.01
    
# limiting the size for the moment because
# with the full set we ran out of memory after training
# on the first epoch
    # trsize = 0.5, tesize = 0.5,

    # use the full dataset
    trsize = None, tesize = None,

    # trsize, tesize = 0.05, 0.05

    # trsize, tesize = 100, 100

    # DEBUG
    # trsize = 0.01, tesize = 0.01,
)


# number of tracks to consider 
numTracksToInclude = 5

#----------------------------------------------------------------------

import trackmodelutils
# make this a global variable for the moment so we can use
# it also in the model building file
tracksVarMaker = trackmodelutils.TrackVarsMaker(numTracksToInclude)


def datasetLoadFunction(fnames, size, cuda, isTraining, reweightPtEta = False):

    assert not reweightPtEta

    data = None

    totsize = 0

    from datasetutils import makeTracksConcatenator, CommonDataConcatenator, SimpleVariableConcatenator, getActualSize

    commonData = CommonDataConcatenator()
    trackVars = makeTracksConcatenator([ 'relpt', 'charge', 'dphiAtVertex', 'detaAtVertex' ] + 
                                       [ 'vtxDz' ])

    otherVars = SimpleVariableConcatenator(['chgIsoWrtChosenVtx', 'chgIsoWrtWorstVtx' ])


    # load all input files
    for fname in fnames:
  
        print "reading",fname
        loaded = torchio.read(fname)

        #----------
        # determine the size
        #----------
        thisSize = getActualSize(size, loaded)

        totsize += thisSize

        #----------
        # combine common data
        #----------
        commonData.add(loaded, thisSize)

        #----------
        # track variables
        #----------
        trackVars.add(loaded, thisSize)

        #----------
        # auxiliary variables
        #----------
        otherVars.add(loaded, thisSize)

        del loaded

    # end of loop over input files

    #----------
    # normalize event weights
    #----------
    commonData.normalizeSignalToBackgroundWeights()

    # make average weight equal to one over the sample
    commonData.normalizeWeights()

    #----------
    data = commonData.data

    #----------
    # convert raw (arbitrary size) track information
    # to variables corresponding to first n tracks
    # in the event
    #
    # we do this here so that we can easily normalize
    # the variables
    #----------
    
    sortedTrackVars = tracksVarMaker.makeVars(dataset = dict(tracks = trackVars.data),
                                              normalizeVars = [ "relpt*",
                                                                "vtxDz*",
                                                                "charge*",
                                                                "detaAtVertex*",
                                                                "dphiAtVertex*",
                                                                ],
                                              trackSelFunc = lambda data, index: data['vtxDz'][index] < 0.01
                                              )

    #----------
    # add track variables
    #----------
    data['sortedTracks'] = sortedTrackVars

    data['otherVars'] = otherVars.data

    data['numTracks'] = np.atleast_2d(
        trackVars.data['numTracks'].astype('float32')
        ).T


    return data, totsize

#----------------------------------------------------------------------
