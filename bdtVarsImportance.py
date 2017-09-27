#!/usr/bin/env python

import re, glob, os, time, tempfile, sys
import numpy as np

import bdtvarsimportanceutils


maxJobsPerGPU = {
    0: 3,
    1: 3,
    }

additionalOptions = [
    ]


import threading

class TrainingRunner(threading.Thread):

    #----------------------------------------

    def __init__(self, outputDir, varnames, excludedVar, useCPU, fomFunction):

        threading.Thread.__init__(self)

        self.outputDir = outputDir
        self.excludedVar = excludedVar
        self.useCPU = useCPU
        self.fomFunction = fomFunction

        # make a copy to be safe
        self.varnames = list(varnames)

    #----------------------------------------

    def setGPU(self, gpuindex):
        self.gpuindex = gpuindex

    #----------------------------------------

    def setGPUmemFraction(self, memFraction):
        self.memFraction = memFraction

    #----------------------------------------

    def setCompletionQueue(self, completionQueue):
        self.completionQueue = completionQueue

    #----------------------------------------

    def setIndex(self, taskIndex):
        self.taskIndex = taskIndex

    #----------------------------------------

    def run(self):
        print "training with",len(self.varnames),",".join(self.varnames)

        # create a temporary dataset file
        text = open(dataSetFname).read()

        dataSetFile = tempfile.NamedTemporaryFile(suffix = ".py", delete = False)

        dataSetFile.write(text)

        print >> dataSetFile,"selectedVariables = ",self.varnames

        dataSetFile.flush()

        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)

        #----------
        # write the variable names to the output directory
        #----------
        fout = open(os.path.join(self.outputDir, "variables.py"), "w")
        print >> fout, self.varnames
        fout.close()

        #----------

        cmdParts = []

        cmdParts.append("./run-gpu.py")

        if self.useCPU:
            cmdParts.append("--gpu cpu")
        else:
            cmdParts.append("--gpu " + str(self.gpuindex))

            if self.memFraction != None:
                cmdParts.append("--memfrac %f" % self.memFraction)

        cmdParts.append("--")

        cmdParts.extend([
            "train01.py",

            # put the dataset specification file first 
            # so that we know the selected variables
            # at the time we build the model
            dataSetFile.name,
            modelFname,
            "--max-epochs " + str(maxEpochs),
            "--output-dir " + self.outputDir,
            ])

        cmdParts.extend(additionalOptions)

        cmd = " ".join(cmdParts)

        res = os.system(cmd)

        if res != 0:
            print "failed to run",cmd
        else:
            print "successfully ran",cmd
        #----------
        # get the results (testAUCs)
        #----------
        testAUC = self.fomFunction(self.outputDir, windowSize)

        result = dict(testAUC = testAUC,
                      varnames = self.varnames,
                      excludedVar = self.excludedVar)
        self.completionQueue.put((self,result))

#----------------------------------------------------------------------

def runTasks(threads, useCPU):
    # runs the given list of tasks on the GPUs and returns
    # when all have finished

    assert len(threads) > 0

    threads = list(threads)

    import Queue as queue
    completionQueue = queue.Queue()

    # maps from GPU number to number of tasks running there
    numThreadsRunning = {}

    for gpu in maxJobsPerGPU.keys():
        numThreadsRunning[gpu] = 0

    #----------
    taskIndex = 0 # for assigning results

    completedTasks = 0

    results = [ None ] * len(threads)

    while completedTasks < len(results):

        # (re)fill queues if necessary

        # distribute jobs equally over GPUs

        # find how many jobs could be started for each GPU
        # do not limit threads when running on CPUs
        while threads:

            # the task to start
            task = None

            if useCPU:
                # we consider to have unlimited amount of CPU cores
                task = threads.pop(0)
            else:
                # running on GPU, check if we have a slot free
                unusedSlots = [ 
                    (maxJobsPerGPU[gpu] - numThreadsRunning[gpu], gpu)
                    for gpu in sorted(numThreadsRunning.keys()) ]

                maxUnusedSlots, maxUnusedGpu = max(unusedSlots)

                if maxUnusedSlots > 0:
                    assert numThreadsRunning[maxUnusedGpu] < maxJobsPerGPU[maxUnusedGpu]
                    task = threads.pop(0)
                    task.setGPU(maxUnusedGpu)

                    # set fraction of GPU memory to use
                    # reduce by some margin, otherwise jobs will not start
                    memMargin = 0.9
                    if task.excludedVar == None:
                        # this is the first (sole) run, use all possible memory 
                        # of one GPU
                        task.setGPUmemFraction(1.0 * memMargin)
                    else:
                        task.setGPUmemFraction(1.0 / float(maxJobsPerGPU[maxUnusedGpu]) * memMargin)

            if task != None:
                task.setIndex(taskIndex)
                taskIndex += 1

                task.setCompletionQueue(completionQueue)

                if useCPU:
                    print "STARTING ON CPU"
                else:
                    numThreadsRunning[maxUnusedGpu] += 1
                    print "STARTING ON GPU",maxUnusedGpu

                task.start()
            else:
                # wait until a task completes
                break

        # wait for any task to complete
        thread, thisResult = completionQueue.get()

        # 'free' a slot on this gpu
        if not useCPU:
            numThreadsRunning[thread.gpuindex] -= 1

        results[thread.taskIndex] = thisResult

        completedTasks += 1

    # end while non completed tasks
        
    return results



#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(prog='bdtVarsImportance.py',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     )

    parser.add_argument('--resume',
                        dest = "resumeDir",
                        metavar = "dir",
                        type = str,
                        default = None,
                        help='resume scan found in the given directory'
                        )

    parser.add_argument('--cpu',
                        dest = "useCPU",
                        default = False,
                        action = 'store_true',
                        help='run on CPU instead of GPUs'
                        )

    bdtvarsimportanceutils.fomAddOptions(parser)

    parser.add_argument('configFile',
                        metavar = "config.py",
                        nargs = 1,
                        help='configuration file'
                        )


    options = parser.parse_args()

    # read parameters for this particular model from the given configuration file
    execfile(options.configFile)

    bdtvarsimportanceutils.fomGetSelectedFunction(options, maxEpochs)

    #----------


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
    # allVars = [ "s4", "scRawE", "scEta", "covIEtaIEta" ]

    allVars = [ "phoIdInput/" + varname for varname in allVars ]

    remainingVars = allVars[:]

    #----------

    results = []

    if options.resumeDir == None:
        # start a new scan
        outputDir = "results/" + time.strftime("%Y-%m-%d-%H%M%S")

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
            
        aucData = bdtvarsimportanceutils.VarImportanceResults()

    else:
        # read results from existing directory
        # assume the set of variables is the same

        outputDir = options.resumeDir
        
        # first find steps which are incomplete and rename these
        # directories
        completeDirs, incompleteDirs = bdtvarsimportanceutils.findComplete(options.resumeDir, maxEpochs)

        for theDir in incompleteDirs.values():
            import shutil

            newName = theDir + "-deleteme"
            index = 1
            while os.path.exists(newName):
                newName = theDir + "-deleteme%d" % index
                index += 1

            print "renaming incomplete directory",theDir,"to",newName
            import shutil
            shutil.move(theDir, newName)

        #----------
        # read the existing results
        #----------

        aucData = bdtvarsimportanceutils.readFromTrainingDir(options.resumeDir, expectedNumEpochs = maxEpochs,
                                                             fomFunction = options.fomFunction)

        # fill into the traditional data structure
        for varnames, testAUC in aucData.data.items():
            results.append(dict(testAUC = testAUC,
                                varnames = varnames))


    # the first number is the round of variable removal
    # the second number is the index corresponding to the variable
    # removed in this round
    jobIndex = [ 0, 0 ]

    #----------
    # run one training with all variables (no variable removed)
    # as a reference
    #----------

    thisOutputDir = os.path.join(outputDir, "%02d-%02d" % tuple(jobIndex))

    # run the training if we don't have the result yet
    if aucData.getOverallAUC() == None:
        thisResults = runTasks([ TrainingRunner(thisOutputDir, allVars, None, options.useCPU, options.fomFunction)], options.useCPU)
    else:
        # take from the existing directory
        thisResults = [ dict(testAUC = aucData.getOverallAUC(),
                             varnames = allVars,
                             excludedVar = None)
                        ]


    results.extend(thisResults)

    # find the one with the highest AUC
    testAUC = thisResults[0]['testAUC']

    print "test AUC of full network:",testAUC

    #----------

    while len(remainingVars) >= 2:
        # eliminate one variable at a time

        jobIndex[0] += 1
        jobIndex[1] = 0

        thisResults = []

        tasks = []

        for excluded in range(len(remainingVars)):

            jobIndex[1] += 1

            thisVars = remainingVars[:excluded] + remainingVars[excluded + 1:]

            thisOutputDir = os.path.join(outputDir, "%02d-%02d" % tuple(jobIndex))

            if aucData.getAUC(thisVars) == None:
                # we need to run this
                tasks.append(TrainingRunner(thisOutputDir, thisVars, remainingVars[excluded], options.useCPU,
                                            options.fomFunction))
            else:
                thisResults.append(dict(testAUC = aucData.getAUC(thisVars),
                                        varnames = thisVars,
                                        excludedVar = remainingVars[excluded]))

        # end of loop over variable to be excluded

        # run the remaining trainings
        if tasks:
            newResults =  runTasks(tasks, options.useCPU)
        else:
            newResults = [] 

        thisResults += newResults
        

        for result in newResults:
            aucData.add(result['varnames'], result['testAUC'])
                              

        for index, line in enumerate(thisResults):
            print "test AUC when removing",line['excludedVar'],"(%d variables remaining)" % (len(remainingVars) - 1),":",line['testAUC']

        sys.stdout.flush()

        results.extend(thisResults)

        # find highest AUC of test data
        # (and variable when removed giving the highest AUC)
        highestAUC, highestAUCvar = max([ (line['testAUC'], line['excludedVar']) for index, line in enumerate(thisResults) 
                                          # avoid removing no variable (because the full
                                          # set of BDT variables typically has the highest AUC)
                                          if line['excludedVar'] != None
                                          ])

        # remove the variable leading to the highest AUC when removed
        print "removing variable",highestAUCvar
        sys.stdout.flush()

        remainingVars.remove(highestAUCvar)

    # end while variables remaining

    print "last remaining variable",remainingVars

    import pickle

    resultFile = os.path.join(outputDir, "results.pkl")
    pickle.dump(results, open(resultFile,"w"))

    print "wrote results to",resultFile


        


    
