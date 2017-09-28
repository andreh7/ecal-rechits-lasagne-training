#!/usr/bin/env python

import re, glob, os, time, tempfile, sys
import numpy as np

import bdtvarsimportanceutils

# use -1 for the CPU
DEVICE_CPU = -1

maxJobsPerGPU = {
    DEVICE_CPU: 4,

    0: 3,
    1: 3,
    }

additionalOptions = [
    ]


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

import threading

class TrainingRunner(threading.Thread):

    #----------------------------------------

    def __init__(self, outputDir, varnames, excludedVar, useCPU, fomFunction, commandPartsBuilder):

        threading.Thread.__init__(self)

        self.outputDir = outputDir
        self.excludedVar = excludedVar
        self.useCPU = useCPU
        self.fomFunction = fomFunction

        # make a copy to be safe
        self.varnames = list(varnames)

        self.commandPartsBuilder = commandPartsBuilder

        self.memFraction = None

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

        # build the command to be run
        cmdParts, logFileName = self.commandPartsBuilder(
            useCPU          = self.useCPU,
            gpuindex        = self.gpuindex,
            memFraction     = self.memFraction,
            dataSetFileName = dataSetFile.name,
            modelFname      = modelFname,
            maxEpochs       = maxEpochs,
            outputDir       = self.outputDir,
            )

        cmdParts.extend(additionalOptions)

        # split command parts with spaces in them
        # (otherwise the command line option parser
        # of the called process will not regonize them, 
        # see https://stackoverflow.com/questions/4091242 )

        tmp = []
        import shlex
        for item in cmdParts:
            tmp.extend(shlex.split(item))
        
        cmdParts = tmp

        #----------
        # start the subprocess
        #----------
        import subprocess

        cmd = " ".join(cmdParts)

        if logFileName != None:
            fout = open(logFileName, "w")
        else:
            fout = None

        res = subprocess.call(cmdParts, stdout = fout, stderr = fout)

        # TODO: is this really needed ?
        fout.close()

        if res != 0:
            print "failed to run",cmd
        else:
            print "successfully ran",cmd
        #----------
        # get the results (testAUCs)
        # note that this may fail if the external command failed
        #----------
        testAUC = None

        try:
            testAUC = self.fomFunction(self.outputDir, windowSize, useBDT = False)
        except Exception, ex:
            print "got exception when getting figure of merit:", str(ex)

        result = dict(testAUC = testAUC,
                      varnames = self.varnames,
                      excludedVar = self.excludedVar,
                      exitStatus = res)
        self.completionQueue.put((self,result))

#----------------------------------------------------------------------

class TasksRunner:

    #----------------------------------------

    def __init__(self, useCPU):

        self.useCPU = useCPU

        # maps from GPU number (-1 for CPU) to number of tasks running there
        self.numThreadsRunning = {}
        
        for device in maxJobsPerGPU.keys():
            # when useCPU is True, do not
            # count slots for GPUs
            if not self.useCPU or device >= 0:
                self.numThreadsRunning[device] = 0

    #----------------------------------------

    def findUnusedDevice(self):
        # @return the device id of the device with the highest
        # number of unused slots (>=0 is GPU, -1 is CPU)
        # or None if all slots are busy

        unusedSlots = [ 
            (maxJobsPerGPU[device] - self.numThreadsRunning[device], device)
            for device in sorted(self.numThreadsRunning.keys()) ]
        

        maxUnusedSlots, maxUnusedDevice = max(unusedSlots)

        if maxUnusedSlots > 0:
            return maxUnusedDevice
        else:
            return None

    #----------------------------------------

    def __setGPUmemFraction(self, task):
        if self.useCPU:
            return

        # reduce by some margin, otherwise jobs will not start
        memMargin = 0.9
        if task.excludedVar == None:
            # this is the first (sole) run, use all possible memory 
            # of one GPU
            task.setGPUmemFraction(1.0 * memMargin)
        else:
            task.setGPUmemFraction(1.0 / float(maxJobsPerGPU[task.gpuindex]) * memMargin)

    #----------------------------------------

    def runTasks(self, threads):

        # runs the given list of tasks on the GPUs and returns
        # when all have finished

        assert len(threads) > 0

        # make a copy
        self.threads = list(threads)

        import Queue as queue
        completionQueue = queue.Queue()


        #----------
        taskIndex = 0 # for assigning results

        completedTasks = 0

        results = [ None ] * len(self.threads)

        # this will be set to true if there is at least one
        # failed task
        drainQueues = False
        numRunningTasks = 0

        while completedTasks < len(results):

            # (re)fill queues if necessary

            # distribute jobs equally over GPUs

            # find how many jobs could be started for each GPU
            # do not limit threads when running on CPUs
            while self.threads:

                # the task to start
                task = None

                # check if we have a slot free
                maxUnusedDevice = self.findUnusedDevice()

                if maxUnusedDevice is not None:

                    assert self.numThreadsRunning[maxUnusedDevice] < maxJobsPerGPU[maxUnusedDevice]
                    task = self.threads.pop(0)
                    task.setGPU(maxUnusedDevice)

                    # set fraction of GPU memory to use
                    self.__setGPUmemFraction(task)
                    
                    task.setIndex(taskIndex)
                    taskIndex += 1

                    task.setCompletionQueue(completionQueue)

                    self.numThreadsRunning[task.gpuindex] += 1

                    if self.useCPU:
                        print "STARTING ON CPU"
                    else:
                        print "STARTING ON GPU",task.gpuindex

                    task.start()
                    numRunningTasks += 1
                else:
                    # wait until a task completes
                    break

            # wait for any task to complete
            thread, thisResult = completionQueue.get()

            # 'free' a slot on this gpu/cpu
            self.numThreadsRunning[thread.gpuindex] -= 1

            results[thread.taskIndex] = thisResult

            completedTasks += 1
            numRunningTasks -= 1

            # check whether the task failed or not
            if thisResult['exitStatus'] != 0:
                drainQueues = True

            if drainQueues and numRunningTasks == 0:
                raise Exception("at least one task had non-zero exit status")

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
    execfile(options.configFile[0])

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
        tasksRunner = TasksRunner(options.useCPU)
        thisResults = tasksRunner.runTasks([ TrainingRunner(thisOutputDir, allVars, None, options.useCPU, options.fomFunction, commandPartsBuilder)])
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
                                            options.fomFunction, commandPartsBuilder))
            else:
                thisResults.append(dict(testAUC = aucData.getAUC(thisVars),
                                        varnames = thisVars,
                                        excludedVar = remainingVars[excluded]))

        # end of loop over variable to be excluded

        # run the remaining trainings
        if tasks:
            tasksRunner = TasksRunner(options.useCPU)
            newResults =  tasksRunner.runTasks(tasks)
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


        


    
