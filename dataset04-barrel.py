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

import numpy as np

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

def datasetLoadFunction(fnames, size, cuda):

    data = None

    totsize = 0

    recHits = RecHitsConcatenator()
  
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
