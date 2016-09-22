# datasets including chose and worst track isolation (2016-05-28)

datasetDir = '../data/2016-05-28-sparse-with-track-iso-vars'

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
    trsize = 0.5,
    tesize = 0.5,

    # trsize, tesize = 0.05, 0.05

    # trsize, tesize = 100, 100

    # DEBUG
    # trsize, tesize = 0.01, 0.01
)
#----------------------------------------

# this is called after loading and combining the given
# input files
def postLoadDataset(label, dataset):
  # normalize in place

  myutils.normalizeVector(dataset.chgIsoWrtChosenVtx)
  myutils.normalizeVector(dataset.chgIsoWrtWorstVtx)

  # DEBUG: just set these values to zero -> we should have the same performance as for the previous training
  # dataset.chgIsoWrtChosenVtx:zero()
  # dataset.chgIsoWrtWorstVtx:zero()

  # checking mean and stddev after normalization

  print label, "chgIsoWrtChosenVtx:", dataset['chgIsoWrtChosenVtx'].mean(), dataset.chgIsoWrtChosenVtx.std()
  print label, "chgIsoWrtWorstVtx:",  dataset['chgIsoWrtWorstVtx'].mean(),  dataset.chgIsoWrtWorstVtx.std() 

#----------------------------------------------------------------------

def datasetLoadFunction(fnames, size, cuda):

    data = None

    totsize = 0

    from datasetutils import makeRecHitsConcatenator, CommonDataConcatenator, SimpleVariableConcatenator, getActualSize

    commonData = CommonDataConcatenator()
    recHits = makeRecHitsConcatenator()

    # some of the BDT input variables
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
        # combine rechits
        #----------
        recHits.add(loaded, thisSize)
    
        #----------
        # auxiliary variables
        #----------
        otherVars.add(loaded, thisSize)
  
    # end of loop over input files

    data = commonData.data
  
    # add rechits
    data['rechits'] = recHits.data

    #----------
    # normalize auxiliary variables to zero mean and unit variance
    #----------
    otherVars.normalize()

    #----------
    # add auxiliary variables
    #----------
    for key, value in otherVars.data.items():
        assert not key in data
        data[key] = value

    assert totsize == data['rechits']['numRecHits'].shape[0]
  
    #----------
    # normalize weights
    #----------
    # to have an average
    # of one per sample
    # (weights should in principle directly
    # affect the effective learning rate of SGD)
    data['weights'] *= (data['weights'].shape[0] / float(data['weights'].sum()))

    return data, totsize

#----------------------------------------------------------------------
