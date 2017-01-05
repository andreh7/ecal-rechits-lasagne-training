#!/usr/bin/env python

import numpy as np

#----------------------------------------------------------------------

isBarrel = True

if globals().has_key('selectedVariables'):
    ninputs = len(selectedVariables)
else:
    if isBarrel:
        ninputs = 12
    else:
        ninputs = 13

#----------------------------------------


#----------------------------------------------------------------------
# function to prepare input data samples
#----------------------------------------------------------------------

def makeInput(dataset, rowIndices, inputDataIsSparse):

    # assert not inputDataIsSparse, "input data is not expected to be sparse"
  
    return [ dataset['input'][rowIndices] ]


#----------------------------------------------------------------------
