#!/usr/bin/env python

#----------------------------------------------------------------------
# utilities for running/checking bdt variable importance scans
#----------------------------------------------------------------------

import os, glob, re

#----------------------------------------------------------------------

class VarImportanceResults:
    # keeps track of test AUCs with a given set of variables

    #----------

    def __init__(self):
        # maps from tuple of variable names
        # to average test AUC
        self.data = {}

        self.allVars = set()

    #----------

    def getOverallAUC(self):
        # returns the AUC with no variable removed
        return self.getAUC(self.allVars)

    #----------
    
    def getAUC(self, varnames):
        return self.data.get(tuple(sorted(varnames)), None)

    #----------
    
    def getTotNumVars(self):
        return len(self.allVars)

    #----------

    def add(self, varnames, testAUC):
        varnames = tuple(sorted(varnames))
        if self.data.has_key(varnames):
            print "WARNING: adding",varnames,"more than once !!"
            
        self.data[varnames] = testAUC

        # update list of all variables seen
        self.allVars = self.allVars.union(varnames)

    #----------------------------------------
        
    def getStepAUCs(self, numVarsRemoved):
        # returns a list of dicts with 
        #   var removed, test AUC with var removed
        
        remainingVars = len(self.allVars) - numVarsRemoved

        results = []

        allVarsThisStep = set()

        for key in self.data.keys():
            if len(key) != remainingVars:
                continue
            results.append(key)

            allVarsThisStep = allVarsThisStep.union(key)

        # now loop again finding out which variable was removed
        retval = []
        for key in results:
            if numVarsRemoved == 0:
                # we train on all variables, there is no variable removed
                removedVariable = None
            else:

                removedVars = allVarsThisStep - set(key)
                assert len(removedVars) == 1,"removedVars is " + str(removedVars)
                removedVariable = list(removedVars)[0]

            retval.append(dict(removedVariable = removedVariable,
                               aucWithVarRemoved = self.data[key]))

        return retval

    #----------------------------------------

    def getNumResultsAtStep(self, numVarsRemoved):

        totNumVars = len(self.allVars)

        # number of variables remaining for this step
        remainingVars = totNumVars - numVarsRemoved

        keys = [ key for key in self.data.keys() if len(key) == remainingVars ]

        return len(keys)

    #----------------------------------------

    def isStepComplete(self, numVarsRemoved):
        # returns true if all steps are present for the give number
        # of removed variables

        # if we remove zero variables, we must have exactly one step
        # if we remove one variable, we must have (totNumVars) steps
        # if we remove two variables, we must have (totNumVars-1) steps
        if numVarsRemoved == 0:
            return self.getOverallAUC() != None

        return self.getNumResultsAtStep(numVarsRemoved) == len(self.allVars) - numVarsRemoved + 1

    #----------------------------------------

    def worstVar(self, numVarsRemoved):
        # return the variable with the highest AUC when removed

        stepAUCs = self.getStepAUCs(numVarsRemoved)

        if stepAUCs:
            maxEl = max(stepAUCs, key = lambda item: item['testAUC']) 
            return maxEl
        else:
            return None

    #----------------------------------------

    def removeVarnamePrefix(self, prefix):
        # removes the given prefix from all
        # variable names which have it
        #
        # only call this when no filling of variables
        # is done afterwards

        def fixVar(varname):
            if varname.startswith(prefix):        
                return varname[len(prefix):]
            else:
                return varname

        newData = {}
        for variables, value in self.data.items():

            newVariables = [ fixVar(varname) for varname in variables]
            newData[tuple(newVariables)] = value

        self.allVars = set( [ fixVar(varname) for varname in self.allVars] )

        self.data = newData

    #----------------------------------------
        


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

def getSigEffAtBgFraction(outputDir, epochs, bgFraction):
    # returns the (averaged) signal fraction at the given background fraction

    import numpy as np

    sigEffs = np.zeros(len(epochs))

    for epochIndex, epoch in enumerate(epochs):

        import plotROCs
        resultDirData = plotROCs.ResultDirData(outputDir, useWeightsAfterPtEtaReweighting = False)

        inputFname = os.path.join(outputDir, "roc-data-test-%04d.npz" % epoch)
        if not os.path.exists(inputFname):
            # try a bzipped version
            inputFname = os.path.join(outputDir, "roc-data-test-%04d.npz.bz2" % epoch)

        auc, numEvents, fpr, tpr = plotROCs.readROC(resultDirData, inputFname, isTrain = False, returnFullCurve = True, updateCache = False)

        # get signal efficiency ('true positive rate' tpr) at given
        # background efficiency ('false positive rate' fpr)

        # assume fpr are sorted so we can use is as 'x value'
        # with function interpolation
        import scipy
        thisSigEff = scipy.interpolate.interp1d(fpr, tpr)(bgFraction)
        sigEffs[epochIndex] = thisSigEff

    # average over the collected iterations
    return sigEffs.mean()

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

        mo = re.match("(\d\d)-(\d\d)$", dirname)
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

def readFromTrainingDir(trainDir, fomFunction = getMeanTestAUC, windowSize = 10, expectedNumEpochs = 200):
    # reads data from the given training directory and
    # returns an object of class VarImportanceResults
    # 
    # @param fomFunction is a function returning the 'figure of merit' (FOM)
    # and takes two arguments theDir and windowSize (how many of the
    # last iterations should be considered)

    retval = VarImportanceResults()

    completeDirs, incompleteDirs = findComplete(trainDir, expectedNumEpochs)

    for theDir in completeDirs.values():
        auc = fomFunction(theDir, windowSize)
        variables = readVars(theDir)
        retval.add(variables, auc)

    return retval

#----------------------------------------------------------------------

def fomAddOptions(parser):
    # adds command line option for known figures of merit

    parser.add_argument('--fom',
                        dest = "fomFunction",
                        type = str,
                        choices = [ 'auc', 
                                    'sigeff005bg',
                                    'sigeff003bg',
                                    'sigeff002bg',
                                    ],
                        default = 'auc',
                        help='figure of merit to use (default: %(default)s)'
                        )

#----------------------------------------------------------------------

def fomGetSelectedFunction(options, expectedNumEpochs):

    if options.fomFunction == 'auc':
        options.fomFunction = getMeanTestAUC
    else:
        mo = re.match('sigeff(\d\d\d)bg', options.fomFunction)

        if mo:
            # signal efficiency at x% fraction of background
            # we specify the epochs explicitly so that we do not 
            # have to read all of them (calculaing the fraction takes some time)

            bgfrac = int(mo.group(1), 10) / 100.0

            # note the +1 because our epoch numbering starts at one
            options.fomFunction = lambda outputDir, windowSize: getSigEffAtBgFraction(outputDir, range(expectedNumEpochs - windowSize + 1, expectedNumEpochs + 1), bgfrac)
        else:
            raise Exception("internal error")

    
