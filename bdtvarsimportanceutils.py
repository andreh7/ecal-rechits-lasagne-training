#!/usr/bin/env python

#----------------------------------------------------------------------
# utilities for running/checking bdt variable importance scans
#----------------------------------------------------------------------

import os, glob, re

def getMeanTestAUC(outputDir, windowSize = 10):

    import numpy as np

    fnames = glob.glob(os.path.join(outputDir, "auc-test-*.txt"))

    epochToAUC = {}

    for fname in fnames:
        mo = re.search("auc-test-(\d+).txt$", fname)
        if not mo:
            continue

        epoch = int(mo.group(1), 10)

        auc = eval(open(fname).read())

        epochToAUC[epoch] = auc

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


