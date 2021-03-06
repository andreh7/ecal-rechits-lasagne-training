#!/usr/bin/env python

import re, glob, os, time, tempfile, sys, signal
import numpy as np

import logging

import bdtvarsimportanceutils

# use -1 for the CPU
DEVICE_CPU = -1

maxJobsPerGPU = {
    # in some test runs with the parallelized
    # BDT code, we had 300% CPU utilization on 
    # average, on a 24 core system we can
    # therefore run ~ 8 trainings in parallel
    DEVICE_CPU: 8,

    0: 3,
    1: 3,
    }

additionalOptions = [
    ]


#----------------------------------------------------------------------

import threading

class TrainingRunner(threading.Thread):

    #----------------------------------------

    def __init__(self, outputDir, varnames, excludedVar, useCPU, fomFunction, resultFileReader, commandPartsBuilder):

        threading.Thread.__init__(self)

        self.outputDir = outputDir
        self.excludedVar = excludedVar
        self.useCPU = useCPU
        self.fomFunction = fomFunction
        self.resultFileReader = resultFileReader

        # make a copy to be safe
        self.varnames = list(varnames)
        self.excludedVar = excludedVar

        self.commandPartsBuilder = commandPartsBuilder

        self.memFraction = None

        self.logger = logging.getLogger("TrainingRunner")

        self.startTime = None
        self.endTime = None

        self.exitStatus = None

        # the result
        self.fom = None

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
        self.startTime = time.time()
        self.logger.info("training with %d variables (%s)", len(self.varnames),",".join(self.varnames))

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

        self.logger.info("starting training: %s", cmd)
        startTime = time.time()
        res = subprocess.call(cmdParts, stdout = fout, stderr = fout)
        elapsed = time.time() - startTime

        # TODO: is this really needed ?
        fout.close()

        if res != 0:
            self.logger.warn("failed to run %s", cmd)
        else:
            self.logger.info("successfully ran %s (%.1f hours wall time)", cmd, elapsed / 3600.)
        #----------
        # get the results (testAUCs)
        # note that this may fail if the external command failed
        #----------
        testAUC = None

        try:
            testAUC = self.fomFunction(self.resultFileReader, self.outputDir, useBDT = False)
        except Exception, ex:
            self.logger.warn("got exception when getting figure of merit: %s", str(ex), exc_info = True)

            # signal problem with calculating figure of merit
            # (in this case we should also stop subsequent jobs)
            res = 256

        self.exitStatus = res

        result = dict(testAUC = testAUC,
                      varnames = self.varnames,
                      excludedVar = self.excludedVar,
                      exitStatus = res)

        self.endTime = time.time()
        self.completionQueue.put((self,result))

#----------------------------------------------------------------------

class TasksRunner:

    #----------------------------------------

    def __init__(self, useCPU):

        self.logger = logging.getLogger("TasksRunner")

        self.useCPU = useCPU

        # maps from GPU number (-1 for CPU) to number of tasks running there
        self.numThreadsRunning = {}
        
        # for keeping a list of running jobs
        self.runningTasks = []
        self.completedTasks = []

        # make a copy
        self.maxJobsPerGPU = {}

        for device in maxJobsPerGPU.keys():
            # when useCPU is True, do not
            # count slots for GPUs
            if self.useCPU and device != DEVICE_CPU:
                continue

            if not self.useCPU and device == DEVICE_CPU:
                continue

            self.maxJobsPerGPU[device] = maxJobsPerGPU[device]
            self.numThreadsRunning[device] = 0

    #----------------------------------------

    def __moveCompletedTaskToCompletedList(self, task):

        for entryIndex in range(len(self.runningTasks)):
            if self.runningTasks[entryIndex] == task:
                self.completedTasks.append(self.runningTasks.pop(entryIndex))
                return

        self.logger.warn("could not find task to be moved from running to completed: %s", str(task))

    #----------------------------------------

    def printTaskStatus(self):
        # prints (using the logging mechanism) the currently
        # running and completed tasks
        #
        # this is typically called on request from the user

        self.logger.info("----------------------------------------")
        self.logger.info("task status at %s", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("----------------------------------------")
        
        now = time.time()

        for description, taskList in (
            ("running", self.runningTasks),
            ("completed", self.completedTasks),
            ):

            self.logger.info("%d %s tasks", len(taskList), description)

            for index, task in enumerate(taskList):
                # can be None
                excludedVar = str(task.excludedVar)

                parts = [ "numVars=%d" % len(task.varnames),
                          "excludedVar=" + excludedVar,
                          "outputDir=" + task.outputDir,
                          ]

                if task.startTime is not None:
                    parts.append("started at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task.startTime)))

                if task.endTime is not None:
                    parts.append("finished at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(task.endTime)))

                if task.startTime is not None:
                    if task.endTime is not None:
                        # completed job
                        parts.append("total wallclock time: %.1f hours" % ((task.endTime - task.startTime) / 3600.))
                    else:
                        # job still running
                        parts.append("elapsed time since start: %.1f hours" % ((now - task.startTime) / 3600.))

                if task.exitStatus is not None:
                    parts.append("exit status %d" % task.exitStatus)

                if task.fom is not None:
                    parts.append("fom=%f" % task.fom)

                self.logger.info("  index %2d: %s", index, ", ".join(parts))

        self.logger.info("----------------------------------------")

    #----------------------------------------

    def findUnusedDevice(self):
        # @return the device id of the device with the highest
        # number of unused slots (>=0 is GPU, -1 is CPU)
        # or None if all slots are busy

        unusedSlots = [ 
            (self.maxJobsPerGPU[device] - self.numThreadsRunning[device], device)
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
            task.setGPUmemFraction(1.0 / float(self.maxJobsPerGPU[task.gpuindex]) * memMargin)

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

                self.logger.debug("found available device %s", str(maxUnusedDevice))
                if maxUnusedDevice is None:
                    self.logger.debug("running tasks: %s", str(self.numThreadsRunning))


                if maxUnusedDevice is not None:

                    assert self.numThreadsRunning[maxUnusedDevice] < self.maxJobsPerGPU[maxUnusedDevice]
                    task = self.threads.pop(0)
                    task.setGPU(maxUnusedDevice)

                    # set fraction of GPU memory to use
                    self.__setGPUmemFraction(task)
                    
                    task.setIndex(taskIndex)
                    taskIndex += 1

                    task.setCompletionQueue(completionQueue)

                    self.numThreadsRunning[task.gpuindex] += 1

                    if self.useCPU:
                        self.logger.info("STARTING ON CPU")
                    else:
                        self.logger.info("STARTING ON GPU",task.gpuindex)

                    self.runningTasks.append(task)

                    task.start()
                    numRunningTasks += 1

                else:
                    # wait until a task completes
                    break

            # wait for any task to complete
            # we put a timeout here so that we can react
            # to SIGUSR1
            while True:
                try:
                    thread, thisResult = completionQueue.get(timeout = 5)
                    break
                except queue.Empty, ex:
                    pass

            # add the figure of merit
            thread.fom = thisResult['testAUC']

            # 'free' a slot on this gpu/cpu
            self.numThreadsRunning[thread.gpuindex] -= 1

            # move tasks to list of completed tasks
            self.__moveCompletedTaskToCompletedList(thread)

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

    #----------
    # setup output directory
    #----------
    if options.resumeDir == None:
        # start a new scan
        outputDir = "results/" + time.strftime("%Y-%m-%d-%H%M%S")

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
    else:
        # resume existing training
        outputDir = options.resumeDir

    #----------
    # setup logging
    #----------
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    logFname = os.path.join(outputDir, "vars-importance-scan.log")

    ch = logging.FileHandler(logFname)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    #----------

    # read parameters for this particular model from the given configuration file
    execfile(options.configFile[0])

    bdtvarsimportanceutils.fomGetSelectedFunction(options, windowSize, maxEpochs)

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
        aucData = bdtvarsimportanceutils.VarImportanceResults()

    else:
        # read results from existing directory
        # assume the set of variables is the same
        
        # first find steps which are incomplete and rename these
        # directories
        completeDirs, incompleteDirs = bdtvarsimportanceutils.findComplete(options.resumeDir, resultFileReader)

        for theDir in incompleteDirs.values():
            import shutil

            newName = theDir + "-deleteme"
            index = 1
            while os.path.exists(newName):
                newName = theDir + "-deleteme%d" % index
                index += 1

            logging.warn("renaming incomplete directory %s to %s",theDir,newName)
            import shutil
            shutil.move(theDir, newName)

        #----------
        # read the existing results
        #----------

        aucData = bdtvarsimportanceutils.readFromTrainingDir(resultFileReader, 
                                                             options.resumeDir, 
                                                             fomFunction = options.fomFunction
                                                             )

        # do NOT fill into results because we will fill
        # results later on with either data from aucData
        # or from new trainings

    # the first number is the round of variable removal
    # the second number is the index corresponding to the variable
    # removed in this round
    jobIndex = [ 0, 0 ]

    #----------
    # run one training with all variables (no variable removed)
    # as a reference
    #----------

    thisOutputDir = os.path.join(outputDir, "%02d-%02d" % tuple(jobIndex))
    
    #----------
    # create the job scheduler object
    tasksRunner = TasksRunner(options.useCPU)

    # install a signal handler to print the job status 
    # whenever we receive SIGUSR1
    signal.signal(signal.SIGUSR1, 
                  lambda signum, frame: tasksRunner.printTaskStatus())

    #----------

    # run the training if we don't have the result yet
    if aucData.getOverallAUC() == None:
        thisResults = tasksRunner.runTasks([ TrainingRunner(thisOutputDir, allVars, None, options.useCPU, options.fomFunction, resultFileReader, commandPartsBuilder)])
    else:
        # take from the existing directory
        thisResults = [ dict(testAUC = aucData.getOverallAUC(),
                             varnames = allVars,
                             excludedVar = None)
                        ]


    results.extend(thisResults)

    # find the one with the highest AUC
    testAUC = thisResults[0]['testAUC']

    logging.info("test FOM with full set of variables: %f",testAUC)

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
                                            options.fomFunction, resultFileReader, commandPartsBuilder))
            else:
                thisResults.append(dict(testAUC = aucData.getAUC(thisVars),
                                        varnames = thisVars,
                                        excludedVar = remainingVars[excluded]))

        # end of loop over variable to be excluded

        # run the remaining trainings
        if tasks:
            newResults =  tasksRunner.runTasks(tasks)
        else:
            newResults = [] 

        thisResults += newResults
        

        for result in newResults:
            aucData.add(result['varnames'], result['testAUC'])
                              

        for index, line in enumerate(thisResults):
            logging.info("test FOM when removing %s (%d variables remaining): %f",
                         str(line['excludedVar']), len(remainingVars) - 1,line['testAUC']
                         )

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
        logging.info("removing variable %s",highestAUCvar)

        sys.stdout.flush()

        remainingVars.remove(highestAUCvar)

    # end while variables remaining

    logging.info("last remaining variable %s",remainingVars)

    import pickle

    resultFile = os.path.join(outputDir, "results.pkl")
    pickle.dump(results, open(resultFile,"w"))

    logging.info("wrote results to %s",resultFile)

        


    
