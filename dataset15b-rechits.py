#!/usr/bin/env python

# same as dataset15-rechits.py but adding tracker isolation variables

execfile("./dataset15-rechits.py")

def datasetLoadFunction(fnames, size, cuda, isTraining, reweightPtEta, logStreams, returnEventIds):
    return __datasetLoadFunctionHelper(fnames, size, cuda, isTraining, reweightPtEta, logStreams, 
                                       returnEventIds, 
                                       additionalVars = ['chgIsoWrtChosenVtx', 'chgIsoWrtWorstVtx' ])

