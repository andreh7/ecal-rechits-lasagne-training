# datasets including chose and worst track isolation (2016-05-28)

# same as dataset04-barrel.py but with newer dataset allowing for pt,eta reweighting
datasetDir = '../data/2016-11-18-photon-et'

isBarrel = True

dataDesc = dict(
    train_files = [ # datasetDir + '/GJet20to40_rechits-barrel-train.t7',
                    datasetDir + '/GJet20toInf_rechits-barrel-train.t7',
                    # datasetDir + '/GJet40toInf_rechits-barrel-train.t7',
                    ],

    test_files  = [ # datasetDir + '/GJet20to40_rechits-barrel-test.t7',
                    datasetDir + '/GJet20toInf_rechits-barrel-test.t7',
                    # datasetDir + '/GJet40toInf_rechits-barrel-test.t7'
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
    trsize = 0.5, tesize = 0.5,

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


def datasetLoadFunction(fnames, size, cuda, isTraining, reweightPtEta = True):

    data = None

    totsize = 0

    from datasetutils import makeRecHitsConcatenator, makeTracksConcatenator, CommonDataConcatenator, SimpleVariableConcatenator, getActualSize, PtEtaReweighter

    commonData = CommonDataConcatenator()
    recHits = makeRecHitsConcatenator()
    trackVars = makeTracksConcatenator([ 'vtxDz' ])

    # only apply pt/eta reweighting for training dataset
    reweightPtEta = reweightPtEta and isTraining

    if reweightPtEta:
      # for reweighting (use reconstructed pt and eta)
      ptEta = SimpleVariableConcatenator(['pt', 'eta'],
                                         dict(pt = lambda loaded:  loaded['phoVars']['phoEt'],
                                              eta = lambda loaded: loaded['phoIdInput']['scEta'])
                                         )

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
        # combine rechits
        #----------
        recHits.add(loaded, thisSize)
    
        #----------
        # track variables
        #----------
        trackVars.add(loaded, thisSize)

        #----------
        # pt/eta reweighting variables
        #----------
        if reweightPtEta:
          ptEta.add(loaded, thisSize)
  
    # end of loop over input files

    #----------
    # reweight signal to have the same background shape
    # using a 2D (pt, eta) histogram
    #----------
    if reweightPtEta:
      ptEtaReweighter = PtEtaReweighter(ptEta.data['pt'][:,0],
                                        ptEta.data['eta'][:,0],
                                        commonData.data['labels'],
                                        isBarrel)

      scaleFactors = ptEtaReweighter.getSignalScaleFactors(ptEta.data['pt'][:,0],
                                                           ptEta.data['eta'][:,0],
                                                           commonData.data['labels'])
      
      # keep original weights
      commonData.data['weightsBeforePtEtaReweighting'] = np.copy(commonData.data['weights'])


      commonData.data['weights'] *= scaleFactors

    #----------
    # normalize event weights
    #----------
    commonData.normalizeSignalToBackgroundWeights()

    # make average weight equal to one over the sample
    commonData.normalizeWeights()

    #----------
    data = commonData.data
  
    # add rechits
    data['rechits'] = recHits.data

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
                                                                "vtxDz*"])

    #----------
    # add track variables
    #----------
    data['sortedTracks'] = sortedTrackVars

    #----------
    # cross check for pt/eta reweighting, dump some variables
    #----------
    if reweightPtEta:
      outputName = "/tmp/pt-reweighting.npz"
      np.savez(outputName,
               pt = ptEta.data['pt'][:,0],
               eta = ptEta.data['eta'][:,0],
               weights = commonData.data['weights'],
               labels = commonData.data['labels'],
               scaleFactors = scaleFactors,
               sigHistogram = ptEtaReweighter.sigHistogram,
               bgHistogram = ptEtaReweighter.bgHistogram,
               )
      print "wrote pt/eta reweighting data to", outputName

      # sys.exit(1)


    assert totsize == data['rechits']['numRecHits'].shape[0]
  
    return data, totsize

#----------------------------------------------------------------------
