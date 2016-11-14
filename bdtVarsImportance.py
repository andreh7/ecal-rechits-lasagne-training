#!/usr/bin/env python

import re, glob, os, time, tempfile, sys
import numpy as np


maxJobsPerGPU = {
    0: 2,
    1: 2,
    }

#----------------------------------------------------------------------

def getMeanTestAUC(outputDir, windowSize = 10):
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
    assert len(epochToAUC) >= windowSize, "have %d in epochToAUC but require %d (windowSize)" % (len(epochToAUC), windowSize)

    aucs = zip(*sorted(epochToAUC.items()))[1]

    # print "aucs=",aucs

    return np.mean(aucs[-windowSize:])

#----------------------------------------------------------------------

import threading

class TrainingRunner(threading.Thread):

    #----------------------------------------

    def __init__(self, outputDir, varnames):

        threading.Thread.__init__(self)

        self.outputDir = outputDir

        # make a copy to be safe
        self.varnames = list(varnames)

    #----------------------------------------

    def setGPU(self, gpuindex):
        self.gpuindex = gpuindex

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
        text = open("dataset06-bdt-inputvars.py").read()

        dataSetFile = tempfile.NamedTemporaryFile(suffix = ".py", delete = False)

        dataSetFile.write(text)

        print >> dataSetFile,"selectedVariables = ",self.varnames

        dataSetFile.flush()

        cmdParts = []

        if self.gpuindex == 0:
            cmdParts.append("./run-gpu0.sh")
        else:
            cmdParts.append("./run-gpu.sh")

        cmdParts.extend([
            "train01.py",

            # put the dataset specification file first 
            # so that we know the selected variables
            # at the time we build the model
            dataSetFile.name,
            "model09-bdt-inputs.py",
            "--max-epochs 50",
            "--output-dir " + self.outputDir,
            ])

        cmd = " ".join(cmdParts)

        res = os.system(cmd)

        if res != 0:
            print "failed to run",cmd

        #----------
        # write the variable names to the output directory
        #----------
        fout = open(os.path.join(self.outputDir, "variables.py"), "w")
        print >> fout, self.varnames
        fout.close()

        #----------
        # get the results (testAUCs)
        #----------
        testAUC = getMeanTestAUC(outputDir, windowSize = 10)

        result = dict(testAUC = testAUC,
                      varnames = self.varnames)
        self.completionQueue.put((self,result))

#----------------------------------------------------------------------

def runTasks(threads):
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
        while threads:
            unusedSlots = [ 
                (maxJobsPerGPU[gpu] - numThreadsRunning[gpu], gpu)
                for gpu in sorted(numThreadsRunning.keys()) ]

            maxUnusedSlots, maxUnusedGpu = max(unusedSlots)

            if maxUnusedSlots > 0:
                assert numThreadsRunning[maxUnusedGpu] < maxJobsPerGPU[maxUnusedGpu]
                task = threads.pop(0)
                task.setGPU(maxUnusedGpu)

                task.setIndex(taskIndex)
                taskIndex += 1

                task.setCompletionQueue(completionQueue)

                numThreadsRunning[maxUnusedGpu] += 1
                print "STARTING ON GPU",maxUnusedGpu
                task.start()

        # wait for any task to complete
        thread, thisResult = completionQueue.get()

        # 'free' a slot on this gpu
        numThreadsRunning[thread.gpuindex] -= 1

        results[thread.taskIndex] = thisResult

        completedTasks += 1

    # end while non completed tasks
        
    return results



#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

if __name__ == '__main__':

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
    thisResults = runTasks([ TrainingRunner(thisOutputDir, allVars)])

    results.extend(thisResults)

    # find the one with the highest AUC
    testAUC = thisResults[0]['testAUC']

    print "test AUC of full network:",testAUC

    #----------

    while len(remainingVars) >= 2:
        # eliminate one variable at a time

        jobIndex[0] += 1
        jobIndex[1] = 0

        tasks = []

        for excluded in range(len(remainingVars)):

            jobIndex[1] += 1

            thisVars = remainingVars[:excluded] + remainingVars[excluded + 1:]

            thisOutputDir = os.path.join(outputDir, "%02d-%02d" % tuple(jobIndex))

            tasks.append(TrainingRunner(thisOutputDir, thisVars))

        # end of loop over variable to be excluded

        # run the trainings
        thisResults = runTasks(tasks)

        for index, line in enumerate(thisResults):
            print "test AUC when removing",remainingVars[index],"(%d variables remaining)" % (len(remainingVars) - 1),":",line['testAUC']

        sys.stdout.flush()

        results.extend(thisResults)

        # find highest AUC of test data
        # (and variable when removed giving the highest AUC)
        highestAUC, highestAUCvarIndex = max([ (line['testAUC'], index) for index, line in enumerate(thisResults) ])

        # remove the variable leading to the highest AUC when removed
        print "removing variable",remainingVars[highestAUCvarIndex]
        sys.stdout.flush()

        del remainingVars[highestAUCvarIndex]

    # end while variables remaining

    print "last remaining variable",remainingVars

    import pickle

    resultFile = os.path.join(outputDir, "results.pkl")
    pickle.dump(results, open(resultFile,"w"))

    print "wrote results to",resultFile


        


    