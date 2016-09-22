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

    from datasetutils import makeRecHitsConcatenator

    recHits = makeRecHitsConcatenator()
  
    # sort the names of the input variables
    # so that we get reproducible results
    sortedVarnames = {}
  
    # load all input files
    for fname in fnames:
  
        print "reading",fname
        loaded = torchio.read(fname)

        #----------
        # determine the size
        #----------
        if size != None and size < 1:
            thisSize = int(size * len(loaded['y']) + 0.5)
        else:
            if size != None:
                thisSize = size
            else:
                thisSize = len(loaded['y'])

            thisSize = min(thisSize, loaded['y'].size[0])

        #----------

        totsize += thisSize

        #----------
        # combine rechits
        #----------
        recHits.add(loaded, thisSize)
    
        if data == None:
    
            #----------
            # create the first entry
            #----------

            data = dict(
               data    = {},
            
               # labels are 0/1 because we use cross-entropy loss
               labels  = loaded['y'].asndarray()[:thisSize].astype('float32'),
            
               weights = loaded['weight'].asndarray()[:thisSize].astype('float32'),
          
               mvaid   = loaded['mvaid'].asndarray()[:thisSize].astype('float32'),
            )

      
            # fill the individual variable names
            sortedVarnames = ['chgIsoWrtChosenVtx', 'chgIsoWrtWorstVtx' ]

            for varname in sortedVarnames:
                # store additional variables by name, not by index
                data[varname] = loaded[varname].asndarray()[:thisSize].astype('float32').reshape((-1,1))
 
        else:

            #----------
            # append
            #----------          

            data['labels']  = np.concatenate((data['labels'],  loaded['y'].asndarray()[:thisSize].astype('float32')))
            data['weights'] = np.concatenate((data['weights'], loaded['weight'].asndarray()[:thisSize].astype('float32')))
            data['mvaid']   = np.concatenate((data['mvaid'],   loaded['mvaid'].asndarray()[:thisSize].astype('float32')))
            
            # concatenate auxiliary variables
            for varname in sortedVarnames:
                data[varname] = np.concatenate([ data[varname], loaded[varname].asndarray()[:thisSize].astype('float32').reshape((-1,1)) ])
          
        # end of appending
  
    # end of loop over input files
  
    # add rechits
    data['rechits'] = recHits.data

    assert totsize == data['rechits']['numRecHits'].shape[0]
  
    #----------
    # normalize weights
    #----------
    # to have an average
    # of one per sample
    # (weights should in principle directly
    # affect the effective learning rate of SGD)
    data['weights'] *= (data['weights'].shape[0] / float(data['weights'].sum()))

    #----------
    # normalize auxiliary variables to zero mean and unit variance
    #----------
    for varname in sortedVarnames:
        data[varname] -= data[varname].mean()

    print "stddevs before:", [ data[varname].std() for varname in sortedVarnames ]

    for varname in sortedVarnames:
        data[varname] /= data[varname].std()

    print "stddevs after:", [ data[varname].std() for varname in sortedVarnames ]

    return data, totsize

#----------------------------------------------------------------------
