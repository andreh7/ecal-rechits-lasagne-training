#!/usr/bin/env python

import re, glob, os, time, tempfile
import numpy as np

#----------------------------------------------------------------------

def doTrain(outputDir, varnames):
    print "training with",len(varnames),",".join(varnames)

    # create a temporary dataset file
    text = open("dataset06-bdt-inputvars.py").read()

    dataSetFile = tempfile.NamedTemporaryFile(suffix = ".py", delete = False)

    dataSetFile.write(text)
    
    print >> dataSetFile,"selectedVariables = ",varnames

    dataSetFile.flush()

    cmdParts = [
        "./run-gpu.sh",
        "train01.py",

        # put the dataset specification file first 
        # so that we know the selected variables
        # at the time we build the model
        dataSetFile.name,
        "model09-bdt-inputs.py",
        "--max-epochs 50",
        "--output-dir " + outputDir,
        ]

    cmd = " ".join(cmdParts)

    res = os.system(cmd)

    if res != 0:
        print "failed to run",cmd

    # get the results
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

    windowSize = 10

    assert len(epochToAUC) >= windowSize, "have %d in epochToAUC but require %d (windowSize)" % (len(epochToAUC), windowSize)

    aucs = zip(*sorted(epochToAUC.items()))[1]

    print "aucs=",aucs

    return dict(testAUC = np.mean(aucs[-windowSize:]))

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

allVars = [
    "s4",
    "scRawE",
    "scEta",
    "covIEtaIEta",
    "rho",
    "pfPhoIso03",
    "phiWidth",
    "covIEtaIPhi",
    "etaWidth",
    # "esEffSigmaRR", # endcap only
    "r9",
    "pfChgIso03",
    "pfChgIso03worst",
    ]

# DEBUG
# allVars = [ "s4", "scRawE" ]

remainingVars = allVars[:]

#----------

outputDir = "results/" + time.strftime("%Y-%m-%d-%H%M%S")

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

results = []

jobIndex = [ 0, 0 ]

#----------
# run one training with all variables (no variable removed)
# as a reference
#----------

thisOutputDir = os.path.join(outputDir, "%02d-%02d" % tuple(jobIndex))
        
# run the training
thisResults = doTrain(thisOutputDir, allVars)

results.append((allVars, thisResults))

# find the one with the highest AUC
testAUC = thisResults['testAUC']

print "test AUC of full network:",testAUC

#----------

while len(remainingVars) >= 2:
    # eliminate one variable at a time

    # AUC of test data
    highestAUC = None
    highestAUCvarIndex = None # variable when removed giving the highest AUC

    jobIndex[0] += 1
    jobIndex[1] = 0

    for excluded in range(len(remainingVars)):

        jobIndex[1] += 1

        thisVars = remainingVars[:excluded] + remainingVars[excluded + 1:]
        
        thisOutputDir = os.path.join(outputDir, "%02d-%02d" % tuple(jobIndex))
        
        # run the training
        thisResults = doTrain(thisOutputDir, thisVars)

        results.append((thisVars, thisResults))

        # find the one with the highest AUC
        testAUC = thisResults['testAUC']

        print "test AUC when removing",remainingVars[excluded],":",testAUC

        if highestAUC == None or testAUC > highestAUC:
            highestAUC = testAUC
            highestAUCvarIndex = excluded

    # end of loop over variable to be excluded

    # remove the variable leading to the highest AUC when removed
    print "removing variable",remainingVars[highestAUCvarIndex]

    del remainingVars[highestAUCvarIndex]

# end while variables remaining

print "last remaining variable",remainingVars

import pickle

resultFile = os.path.join(outputDir, "results.pkl")
pickle.dump(results, open(resultFile,"w"))

print "wrote results to",resultFile


        


    
