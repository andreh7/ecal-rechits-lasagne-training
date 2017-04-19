#!/usr/bin/env python

#----------------------------------------------------------------------
# utilities for running/checking bdt variable importance scans
#----------------------------------------------------------------------

import os, glob, re

#----------------------------------------------------------------------

def getAUCs(outputDir, sample = "test"):
    # returns a dict of epoch number to AUC
    
    fnames = glob.glob(os.path.join(outputDir, "auc-" + sample + "-*.txt"))

    epochToAUC = {}

    for fname in fnames:
        mo = re.search("auc-test-(\d+).txt$", fname)
        if not mo:
            continue

        epoch = int(mo.group(1), 10)

        auc = eval(open(fname).read())

        epochToAUC[epoch] = auc

    return epochToAUC

#----------------------------------------------------------------------

def isComplete(outputDir, numEpochs, sample = "test"):

    epochToAUC = getAUCs(outputDir, sample)

    # check that we have exactly 1..numEpochs as keys

    if len(epochToAUC) != numEpochs:
        return False

    return all(epochToAUC.has_key(epoch) for epoch in range(1, numEpochs + 1))

#----------------------------------------------------------------------

def getMeanTestAUC(outputDir, windowSize = 10):

    import numpy as np

    epochToAUC = getAUCs(outputDir, "test")

    # average over the last few iterations
    assert len(epochToAUC) >= windowSize, "have %d in epochToAUC but require %d (windowSize) in directory %s" % (len(epochToAUC), windowSize, outputDir)

    aucs = zip(*sorted(epochToAUC.items()))[1]

    # print "aucs=",aucs

    return float(np.mean(aucs[-windowSize:]))

#----------------------------------------------------------------------

def readVars(dirname):
    fin = open(os.path.join(dirname,"variables.py"))
    retval = eval(fin.read())
    fin.close()
    return retval

#----------------------------------------------------------------------

def findComplete(trainDir, expectedNumEpochs = 200):
    # returns (map of subdirectories with the complete number of epochs),
    #         (map of subdirectories with good name but not complete)
    #
    # keys are (index, subindex), values are directory names
    #
    completeDirs = {}
    incompleteDirs = {}

    for dirname in os.listdir(trainDir):

        mo = re.match("(\d\d)-(\d\d)", dirname)
        if not mo:
            continue

        index    = int(mo.group(1),10)
        subindex = int(mo.group(2),10)

        fullPath = os.path.join(trainDir, dirname)

        if not os.path.isdir(fullPath):
            continue

        # check if this is complete
        if isComplete(fullPath, expectedNumEpochs):
            completeDirs[(index, subindex)] = fullPath
        else:
            incompleteDirs[(index, subindex)] = fullPath

    return completeDirs, incompleteDirs

#----------------------------------------------------------------------

def readFromTrainingDir(trainDir):
    # reads from the training directory

    # read initial training
    overallDir = os.path.join(trainDir, "00-00")
    overallAUC = getMeanTestAUC(overallDir)
    allVars = readVars(overallDir)

    import itertools

    stepData = []

    allVarsCurrentStep = list(allVars)

    for index in itertools.count(1):
        thisStep = dict(
            aucWithVarRemoved = {},
            )

        remainingVars = None
        allVariablesOrdered = []

        for subIndex in itertools.count(1):
            # read variables of this step
            
            inputDir = os.path.join(trainDir,
                                    "%02d-%02d" % (index, subIndex))

            if not os.path.exists(inputDir):
                break

            thisVars = readVars(inputDir)

            if remainingVars == None:
                remainingVars = len(thisVars)
                thisStep['numRemainingVars'] = remainingVars
            else:
                assert remainingVars == len(thisVars)

            #----------
            # find which variable was removed
            #----------
            assert len(thisVars) + 1 == len(allVarsCurrentStep)

            removedVar = None
            
            for var in allVarsCurrentStep:
                if not var in thisVars:
                    removedVar = var
                    break

            assert removedVar != None

            allVariablesOrdered.append(removedVar)

            # read AUC
            thisAUC = getMeanTestAUC(inputDir)
            
            thisStep['aucWithVarRemoved'][removedVar] = thisAUC


        # end of loop over subindices

        if subIndex == 1:
            # no directory for this index found
            break

        # end of loop over subIndex

        #----------
        # complete some more information
        #----------
        thisStep['bestAUC'], varToRemove = max( [ (auc, var) for var, auc in thisStep['aucWithVarRemoved'].items() ] )
        thisStep['removedVariable'] = varToRemove
        thisStep['allVariables'] = allVariablesOrdered

        stepData.append(thisStep)

        # prepare next iteration
        allVarsCurrentStep.remove(varToRemove)

    # end of loop over indices

    return stepData, overallAUC


#----------------------------------------------------------------------


