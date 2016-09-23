#!/usr/bin/env python

# datasets with BDT input variables

import math, torchio

datasetDir = '../data/2016-07-06-bdt-inputs'

dataDesc = dict(

    train_files = [ datasetDir + '/GJet20to40_rechits-barrel-train.t7',
                    datasetDir + '/GJet20toInf_rechits-barrel-train.t7',
                    datasetDir + '/GJet40toInf_rechits-barrel-train.t7',
                    ],

    test_files  = [ datasetDir + '/GJet20to40_rechits-barrel-test.t7',
                    datasetDir + '/GJet20toInf_rechits-barrel-test.t7',
                    datasetDir + '/GJet40toInf_rechits-barrel-test.t7',
                    ],

    inputDataIsSparse = False,

    # if one specifies nothing (or None), the full sizes
    # from the input samples are taken

    #  if one specifies values < 1 these are interpreted
    #  as fractions of the sample
    #  trsize, tesize = 10000, 1000
    #  trsize, tesize = 0.1, 0.1
    #  trsize, tesize = 0.01, 0.01
    # 
    #  limiting the size for the moment because
    #  with the full set we ran out of memory after training
    #  on the first epoch
    #  trsize, tesize = 0.5, 0.5
    
    trsize = None, 
    tesize = None,


    # DEBUG
    # trsize, tesize = 0.01, 0.01
    # trsize, tesize = 100, 100
)   
#----------------------------------------


### -- this is called after loading and combining the given
### -- input files
### function postLoadDataset(label, dataset)
### 
### end

#--------------------------------------
# input variables
#--------------------------------------
#
#   phoIdInput :
#     {
#       covIEtaIEta : FloatTensor - size: 431989          0
#       covIEtaIPhi : FloatTensor - size: 431989          1
#       esEffSigmaRR : FloatTensor - size: 431989         2 # endcap only
#       etaWidth : FloatTensor - size: 431989             3
#       pfChgIso03 : FloatTensor - size: 431989           4
#       pfChgIso03worst : FloatTensor - size: 431989      5
#       pfPhoIso03 : FloatTensor - size: 431989           6
#       phiWidth : FloatTensor - size: 431989             7
#       r9 : FloatTensor - size: 431989                   8
#       rho : FloatTensor - size: 431989                  9
#       s4 : FloatTensor - size: 431989                  10
#       scEta : FloatTensor - size: 431989               11
#       scRawE : FloatTensor - size: 431989              12
#     }

#----------------------------------------------------------------------

def datasetLoadFunction(fnames, size, cuda):

    data = None

    totsize = 0

    from datasetutils import makeRecHitsConcatenator, CommonDataConcatenator, SimpleVariableConcatenator, getActualSize

    commonData = CommonDataConcatenator()
  
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
        thisSize = getActualSize(size, loaded)

        totsize += thisSize

        #----------
        # combine common data
        #----------
        commonData.add(loaded, thisSize)

    
        if data == None:
    
            #----------
            # create the first entry
            #----------
      
            # fill the individual variable names
            sortedVarnames = sorted(loaded['phoIdInput'].keys())

            # for barrel, exclude the preshower variable
            # (this will have zero standard deviation in the barrel and will
            # therefore lead to NaNs after normalization)
            sortedVarnames = [ varname for varname in sortedVarnames if varname != 'esEffSigmaRR' ]

            numvars = len(sortedVarnames)
      
            # allocate a 2D Tensor
            data['input'] = np.ndarray((thisSize, numvars), dtype = 'float32')
      
            # copy over the individual variables: use a 2D tensor
            # with each column representing a variables
            
            for varindex, varname in enumerate(sortedVarnames):

              data['input'][:, varindex] = loaded['phoIdInput'][varname].asndarray()[:thisSize]
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
            newData = np.ndarray((thisSize, numvars), dtype = 'float32')
      
            for varindex, varname in enumerate(sortedVarnames):
                newData[:,varindex] = loaded['phoIdInput'][varname].asndarray()[:thisSize]
      
            # and append
            data['input']    = np.concatenate((data['input'], newData))
          
        # end of appending
  
    # end of loop over input files
  
    # TODO: do we actually use this ?
    data['size'] = lambda: totsize 
    
    assert totsize == data['input'].shape[0]
  
    # normalize weights to have an average
    # of one per sample
    # (weights should in principle directly
    # affect the effective learning rate of SGD)
    data['weights'] *= (data['weights'].shape[0] / float(data['weights'].sum()))

    # normalize inputs to zero mean and unit variance
    data['input'] -= data['input'].mean(axis = 0)

    print "stddevs before:",data['input'].std(axis = 0)

    data['input'] /= data['input'].std(axis = 0)
  
    print "stddevs after:",data['input'].std(axis = 0)

    return data, totsize

#----------------------------------------------------------------------
