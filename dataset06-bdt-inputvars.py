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
#       s4 : FloatTensor - size: 431989
#       scRawE : FloatTensor - size: 431989
#       scEta : FloatTensor - size: 431989
#       covIEtaIEta : FloatTensor - size: 431989
#       rho : FloatTensor - size: 431989
#       pfPhoIso03 : FloatTensor - size: 431989
#       phiWidth : FloatTensor - size: 431989
#       covIEtaIPhi : FloatTensor - size: 431989
#       etaWidth : FloatTensor - size: 431989
#       esEffSigmaRR : FloatTensor - size: 431989
#       r9 : FloatTensor - size: 431989
#       pfChgIso03 : FloatTensor - size: 431989
#       pfChgIso03worst : FloatTensor - size: 431989
#     }

#----------------------------------------------------------------------

def datasetLoadFunction(fnames, size, cuda):

    data = None

    totsize = 0
  
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
            thisSize = math.floor(size * len(loaded['y']) + 0.5)
        else:
            if size != None:
                thisSize = size
            else:
                thisSize = len(loaded['y'])

            thisSize = min(thisSize, loaded['y'].size[0])

        #----------

        totsize += thisSize
    
        if data == None:
    
            #----------
            # create the first entry
            #----------

            data = dict(
               data    = {},
            
               # labels are 0/1 because we use cross-entropy loss
               labels  = loaded['y'].asndarray()[:thisSize],
            
               weights = loaded['weight'].asndarray()[:thisSize],
          
               mvaid   = loaded['mvaid'].asndarray()[:thisSize],
            )
      
            # fill the individual variable names
            sortedVarnames = sorted(loaded['phoIdInput'].keys())
            numvars = len(sortedVarnames)
      
            # allocate a 2D Tensor
            data['input'] = np.ndarray((thisSize, numvars))
      
            # copy over the individual variables: use a 2D tensor
            # with each column representing a variables
            
            for varindex, varname in enumerate(sortedVarnames):

              data['input'][:, varindex] = loaded['phoIdInput'][varname].asndarray()[:thisSize]
        else:

            #----------
            # append
            #----------          
            
            # see http://stackoverflow.com/a/36242627/288875 for why one has to
            # put the arguments in parentheses...
            data['labels']  = np.concatenate((data['labels'], loaded['y'].asndarray()[:thisSize]))
    
            data['weights'] = np.concatenate((data['weights'], loaded['weight'].asndarray()[:thisSize]))
    
            data['mvaid']   = np.concatenate((data['mvaid'], loaded['mvaid'].asndarray()[:thisSize]))
    
            # special treatment for input variables
      
            # note that we can not use resize(..) here as the contents
            # of the resized tensor are undefined according to 
            # https://github.com/torch/torch7/blob/master/doc/tensor.md#resizing
            #
            # so we build first a tensor with the new values
            # and then concatenate this to the previously loaded data
            newData = np.ndarray((thisSize, numvars))
      
            for varindex, varname in enumerate(sortedVarnames):
                newData[:,varindex] = loaded['phoIdInput'][varname].asndarray()[:thisSize]
      
            # and append
            data['input']    = np.concatenate((data['input'], newData))
          
        # end of appending
  
    # end of loop over input files
  
    ### ----------
    ### -- convert to CUDA tensors if required
    ### ----------
    ### if cuda then
    ###   data.labels  = data.labels:cuda()
    ###   data.weights = data.weights:cuda()
    ###   data.mvaid   = data.mvaid:cuda()
    ### end
    ### 
    ### ----------
  
    # TODO: do we actually use this ?
    data['size'] = lambda: totsize 
    
    assert totsize == data['input'].shape[0]
  
    # normalize weights to have an average
    # of one per sample
    # (weights should in principle directly
    # affect the effective learning rate of SGD)
    data['weights'] *= (data['weights'].shape[0] / float(data['weights'].sum()))
  
    return data, totsize

#----------------------------------------------------------------------
