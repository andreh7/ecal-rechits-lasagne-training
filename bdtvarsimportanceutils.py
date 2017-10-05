#!/usr/bin/env python

#----------------------------------------------------------------------
# utilities for running/checking bdt variable importance scans
#----------------------------------------------------------------------

import os, glob, re, logging

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

def getMeanTestAUC(resultFileReader, outputDir, useBDT):
    # relies on the fact that the ResultFileReader only
    # looks at the test data (not the train data)

    assert not useBDT

    import numpy as np

    rocDatas = resultFileReader.getROCs(outputDir, useBDT)

    meanAUC = np.mean([ rocData['auc'] for rocData in rocDatas ])

    return float(meanAUC)

#----------------------------------------------------------------------
# classes to return a list of file names to be read 
# and return the corresponding data (true and false positive
# rate arrays etc.) on which a figure of merit
# can be calculated

class ResultFileReaderNN:

    def __init__(self, windowSize, expectedNumEpochs):

        # caculate the epoch numbers
        # (we will not use them if useBDT is True)
        self.epochs = range(expectedNumEpochs - windowSize + 1, expectedNumEpochs + 1)

    #----------------------------------------

    def __getListOfFiles(self, outputDir, useBDT):

        if useBDT:
            # there are no epochs
            result = [ "roc-data-test-mva.npz" ]
            
        else:
            result = [ "roc-data-test-%04d.npz" % epoch 
                       for epoch in self.epochs ]

        # add directory name
        return [ os.path.join(outputDir, fname)
                 for fname in result ]

    #----------------------------------------

    def getROCs(self, outputDir, useBDT):
        # @return a list of dicts with auc, numEvents, fpr, tpr
        #
        # @param useBDT if True, returns the figure of merit
        # of the official photon id values (no averaging 
        # over epochs since we only have one 

        fnames = self.__getListOfFiles(outputDir, epochs)
        
        import plotROCs
        resultDirData = plotROCs.ResultDirData(outputDir, useWeightsAfterPtEtaReweighting = False)

        from ResultDirRocs import ResultDirRocs
        resultDirRocs = ResultDirRocs(resultDirData, maxNumThreads = None)

        result = []

        for inputFname in fnames:
            
            if not os.path.exists(inputFname):
                # try a bzipped version
                inputFname = os.path.join(outputDir, namingFunc(epoch) + ".bz2")

            auc, numEvents, fpr, tpr = resultDirRocs.readROC(inputFname, isTrain = False, returnFullCurve = True, updateCache = False)

            result.append(dict(
                    inputFname = inputFname,
                    auc = auc,
                    numEvents = numEvents,
                    fpr = fpr,
                    tpr = tpr
                    ))
                    
        return result

    #----------------------------------------

    def isComplete(self, outputDir):
        # @return true if the given output directory is considered to be complete
        # (this is used for resuming importance scan sessions)
        
        sample = "test"

        fnames = glob.glob(os.path.join(outputDir, "auc-" + sample + "-*.txt"))

        epochs = set()

        for fname in fnames:
            mo = re.search("auc-test-(\d+).txt$", fname)
            if mo:
                epoch = int(mo.group(1), 10)
                epochs.add(epoch)

        # check that we have exactly 1..numEpochs as keys
        return epochs == set(self.epochs)


#----------------------------------------------------------------------

class ResultFileReaderTMVA:
    # reading results from a TMVA .root output file

    def __init__(self):
        # we do not need to average over epochs here
        # but we need to know which variable we should look
        # for in the test tree

        self.logger = logging.getLogger("ResultFileReaderTMVA")

    #----------------------------------------

    def getROCs(self, outputDir, useBDT):
        # @return a list of dicts with auc, numEvents, fpr, tpr
        #
        # @param useBDT if True, returns the figure of merit
        # of the official photon id values (no averaging 
        # over epochs since we only have one 

        inputFname = os.path.join(outputDir, "tmva.root")

        import plotROCsTMVA

        result = []

        # TODO: make order of returned values of this readROC() function the same
        #       as the one in plotROCs
        auc, fpr, tpr, numEvents = plotROCsTMVA.readROC(inputFname, isTrain = False, returnFullCurve = True, origMVA = useBDT,
                                                        logger = self.logger)

        result.append(dict(
                inputFname = inputFname,
                auc = auc,
                numEvents = numEvents,
                fpr = fpr,
                tpr = tpr
                ))
        
        return result

    #----------------------------------------

    def isComplete(self, outputDir):
        # basically insist that there is the .xml weights
        # file and the .root output file
        #
        # in principle we should also ensure that the .root file
        # has been closed properly..

        for fname in ('tmva.root', 
                      'TMVAClassification_BDT.weights.xml'):
            if not os.path.exists(
                os.path.join(outputDir, fname)):
                return False

        return True

#----------------------------------------------------------------------


class SigEffAtBgFractionFunc:
    # functor which returns the (averaged) signal fraction at the given background fraction
    # 
    # note that we make this a class instead of a function 
    # so that it can be pickled which we need when
    # using the multiprocessing module

    #----------------------------------------

    def __init__(self, bgFraction):
        self.bgFraction = bgFraction

    #----------------------------------------

    def __call__(self, resultFileReader, outputDir, useBDT):
        # @param useBDT if True, returns the figure of merit
        # of the official photon id values (no averaging 
        # over epochs since we only have one 

        import numpy as np

        rocDatas = resultFileReader.getROCs(outputDir, useBDT)
        sigEffs = np.zeros(len(rocDatas))

        for index, rocData in enumerate(rocDatas):

            # get signal efficiency ('true positive rate' tpr) at given
            # background efficiency ('false positive rate' fpr)

            # assume fpr are sorted so we can use is as 'x value'
            # with function interpolation
            import scipy.interpolate
            thisSigEff = scipy.interpolate.interp1d(rocData['fpr'], rocData['tpr'])(self.bgFraction)
            sigEffs[index] = thisSigEff

        # average over the collected iterations
        return sigEffs.mean()

#----------------------------------------------------------------------

def readVars(dirname):
    fin = open(os.path.join(dirname,"variables.py"))
    retval = eval(fin.read())
    fin.close()
    return retval

#----------------------------------------------------------------------

def findComplete(trainDir, resultFileReader):
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
        if resultFileReader.isComplete(fullPath):
            completeDirs[(index, subindex)] = fullPath
        else:
            incompleteDirs[(index, subindex)] = fullPath

    return completeDirs, incompleteDirs

#----------------------------------------------------------------------

class __ReadFromTrainingDirHelperFunc:
    def __init__(self, func, resultFileReader):
        self.func = func
        self.resultFileReader = resultFileReader

    def __call__(self, theDir):
        return self.func(resultFileReader, theDir, useBDT = False)

def readFromTrainingDir(resultFileReader, trainDir, fomFunction,
                        numParallelProcesses = None):
    # reads data from the given training directory and
    # returns an object of class VarImportanceResults
    # 
    # @param fomFunction is a function returning the 'figure of merit' (FOM)
    # and takes two arguments theDir and resultFileReader

    retval = VarImportanceResults()

    completeDirs, incompleteDirs = findComplete(trainDir, resultFileReader)

    if numParallelProcesses == None:
        aucs = [ fomFunction(resultFileReader, theDir, useBDT = False) for theDir in completeDirs.values() ]
    else:
        import multiprocessing
        if numParallelProcesses >= 1:
            pool = multiprocessing.Pool(processes = numParallelProcesses)
        else:
            pool = multiprocessing.Pool()

        # need a pickleable object
        helperFunc = __ReadFromTrainingDirHelperFunc(fomFunction, resultFileReader)

        aucs = pool.map(helperFunc, completeDirs.values())

    for theDir, auc in zip(completeDirs.values(), aucs):
        variables = readVars(theDir)
        retval.add(variables, auc)

    return retval

#----------------------------------------------------------------------

def readFromResultsFile(resultsFname):
    # fills an VarImportanceResults() object with data from
    # a results.pkl file

    retval = VarImportanceResults()
    
    import cPickle as pickle
    
    results = pickle.load(open(resultsFname))

    for line in results:
        retval.add(line['varnames'], line['testAUC'])

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


def fomGetSelectedFunction(options, windowSize, expectedNumEpochs):

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
            options.fomFunction = SigEffAtBgFractionFunc(bgfrac)
        else:
            raise Exception("internal error")

#----------------------------------------------------------------------

def commandPartsBuilderNN(useCPU,
                          gpuindex,
                          memFraction,
                          dataSetFileName,
                          modelFname,
                          maxEpochs,
                          outputDir,
                          ):
    # returns a list of command parts for running neural network trainings
    cmdParts = []

    cmdParts.append("./run-gpu.py")

    if useCPU:
        cmdParts.append("--gpu cpu")
    else:
        cmdParts.append("--gpu " + str(gpuindex))

        if memFraction != None:
            cmdParts.append("--memfrac %f" % memFraction)

    cmdParts.append("--")

    cmdParts.extend([
        "train01.py",

        # put the dataset specification file first 
        # so that we know the selected variables
        # at the time we build the model
        dataSetFileName,
        modelFname,
        "--max-epochs " + str(maxEpochs),
        "--output-dir " + outputDir,
        ])

    return cmdParts, None

#----------------------------------------------------------------------

def commandPartsBuilderBDT(useCPU,
                           gpuindex,
                           memFraction,
                           dataSetFileName,
                           modelFname,
                           maxEpochs,
                           outputDir,
                           ):
    # returns a list of command parts for running neural network trainings
    cmdParts = []

    cmdParts.extend([
        "./train-tmva.py",

        # put the dataset specification file first 
        # so that we know the selected variables
        # at the time we build the model
        dataSetFileName,
        modelFname,
        "--output-dir " + outputDir,
        ])

    logFile = os.path.join(outputDir, "tmva-training.log")
    
    return cmdParts, logFile

#----------------------------------------------------------------------

