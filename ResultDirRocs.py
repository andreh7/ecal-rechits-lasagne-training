#!/usr/bin/env python

#----------------------------------------------------------------------

import glob, os, re, sys

#----------------------------------------------------------------------

def _readROCfilesLambda(fname, isTrain):
    # default transformation function for readROCfiles
    return fname

#----------------------------------------------------------------------

class Func:
    # function wrapper used with the multiprocessing module
    def __init__(self, func):
        self.func = func

    def __call__(self, args):
        return self.func(*args)

class ReadROChelper:
    def __init__(self, resultDirRocs):
        self.resultDirRocs = resultDirRocs
        
    def __call__(self, *args):
        return self.resultDirRocs.readROC(*args)

#----------------------------------------------------------------------

class ResultDirRocs:
    """ caches ROC values from a result directory """
    #----------------------------------------

    def __init__(self, resultDirData, minEpoch = None, maxEpoch = None, 
                 excludedEpochs = None,
                 maxNumThreads = 8):
        # to keep weights
        self.resultDirData = resultDirData
        self.minEpoch = minEpoch
        self.maxEpoch = maxEpoch
        self.excludedEpochs = excludedEpochs

        self.maxNumThreads = maxNumThreads

        # read only the file names
        self.mvaROCfnames, self.rocFnames = self.readROCfiles()

    #----------------------------------------

    def readROCfiles(self, transformation = None, includeCached = False):
        # returns mvaROC, rocValues
        # which are dicts of 'test'/'train' to the single value
        # (for MVAid) or a dict epoch -> values (rocValues)
        #
        # transformation is a function taking the file name 
        # which is run on each file
        # found and stored in the return values. If None,
        # just the name is stored.

        if transformation == None:
            transformation = _readROCfilesLambda

        inputDir = self.resultDirData.inputDir

        #----------
        inputFiles = []

        if includeCached:
            # read cached version first
            inputFiles += glob.glob(os.path.join(inputDir, "roc-data-*.npz.cached-auc.py")) 

        inputFiles += glob.glob(os.path.join(inputDir, "roc-data-*.npz.bz2")) 
        inputFiles += glob.glob(os.path.join(inputDir, "roc-data-*.npz")) 

        if not inputFiles:
            print >> sys.stderr,"no files roc-data-* found, exiting"
            sys.exit(1)

        # ROCs values and epoch numbers for training and test
        # first index is 'train or 'test'
        # second index is epoch number
        rocValues    = dict(train = {}, test = {})

        # MVA id ROC areas
        mvaROC = dict(train = None, test = None)

        tasks = []

        # to avoid scheduling tasks twice (because the result
        # is not present yet), in particular avoid re-reading
        # the full file even though the cached file is present
        scheduledTasks = dict(train = set(), test = set())

        for inputFname in inputFiles:

            basename = os.path.basename(inputFname)

            # example names:
            #  roc-data-test-mva.npz
            #  roc-data-train-0002.npz

            mo = re.match("roc-data-(\S+)-mva\.npz(\.bz2)?$", basename)

            if mo:
                sampleType = mo.group(1)

                assert mvaROC.has_key(sampleType)
                assert mvaROC[sampleType] == None

                isTrain = sampleType == 'train'

                mvaROC[sampleType] = transformation(inputFname, isTrain)

                continue

            mo = re.match("roc-data-(\S+)-(\d+)\.npz(\.bz2)?$", basename)

            if not mo and includeCached:
                mo = re.match("roc-data-(\S+)-(\d+)\.npz\.cached-auc\.py$", basename)

            if mo:
                sampleType = mo.group(1)
                epoch = int(mo.group(2), 10)
                isTrain = sampleType == 'train'

                if (self.minEpoch == None or epoch >= self.minEpoch) and (self.maxEpoch == None or epoch <= self.maxEpoch):
                    if self.excludedEpochs == None or not epoch in self.excludedEpochs:

                        if rocValues[sampleType].has_key(epoch):
                            # skip reading this file: we already have a value
                            # (priority is given to the cached files)
                            continue

                        if epoch in scheduledTasks[sampleType]:
                            # already scheduled, no need to schedule twice
                            continue

                    tasks.append(dict(
                            sampleType = sampleType,
                            epoch = epoch,
                            args = (inputFname, isTrain),
                            ))
                    scheduledTasks[sampleType].add(epoch)

                continue

            print >> sys.stderr,"WARNING: unmatched filename",inputFname

        # calculate the AUC values
        from multiprocessing import Process, Pool

        if self.maxNumThreads != None:
            # multiprocessing enabled
            procPool = Pool(processes = self.maxNumThreads)

            results = procPool.map(Func(transformation), [ task['args'] for task in tasks ])

            # wait for processes to complete
            procPool.close()
            procPool.join()

        else:
            # run in the current thread only
            results = [ Func(transformation)(task['args']) for task in tasks ]

        for task, res in zip(tasks, results):
            rocValues[task['sampleType']][task['epoch']] = res

        return mvaROC, rocValues

    #----------------------------------------

    def findLastCompleteEpoch(self, ignoreTrain):

        trainEpochNumbers = sorted(self.rocFnames['train'].keys())
        testEpochNumbers  = sorted(self.rocFnames['test'].keys())

        if not ignoreTrain and not trainEpochNumbers:
            print >> sys.stderr,"WARNING: no training files found"
            return None

        if not testEpochNumbers:
            print >> sys.stderr,"WARNING: no test files found"
            return None

        retval = testEpochNumbers[-1]

        if not ignoreTrain:
            retval = min(trainEpochNumbers[-1], retval)
        return retval

    #----------------------------------------

    def readROC(self, fname, isTrain, returnFullCurve = False, updateCache = True):
        # reads a torch/npz file and calculates the area under the ROC
        # curve for it
        # 
        # also looks for a cached file

        if fname.endswith(".cached-auc.py"):
            if returnFullCurve:
                raise Exception("returnFullCurve is not supported when reading cached AUC files")

            # read the cached file
            print "reading",fname
            auc = float(open(fname).read())
            return auc

        print "reading",fname

        assert fname.endswith(".npz") or fname.endswith(".npz.bz2")
        try:
            import numpy as np
            if fname.endswith(".npz.bz2"):
                import bz2
                data = np.load(bz2.BZ2File(fname))
            else:
                data = np.load(fname)
        except Exception, ex:
            raise Exception("error caught reading " + fname, ex)

        weights = self.resultDirData.getWeights(isTrain)
        labels  = self.resultDirData.getLabels(isTrain)
        outputs = data['output']

        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(labels, outputs, sample_weight = weights)

        aucValue = auc(fpr, tpr, reorder = True)

        #----------

        if updateCache:
            # write to cache
            cachedFname = fname + ".cached-auc.py"
            fout = open(cachedFname,"w")
            print >> fout,aucValue
            fout.close()

            # also copy the timestamp so that we can 
            # use it for estimating the time elapsed
            # for the plot
            modTime = os.path.getmtime(fname)
            os.utime(cachedFname, (modTime, modTime))

        #----------

        if returnFullCurve:
            return aucValue, len(weights), fpr, tpr, thresholds
        else:
            return aucValue

    #----------------------------------------


    def __getInputFname(self, epoch, isTrain):
        # epoch can also be 'BDT', otherwise a number

        if isTrain:
            sample = 'train'
        else:
            sample = 'test'


        if epoch == 'BDT':
            return self.mvaROCfnames[sample]
        else:
            return self.rocFnames[sample][epoch]

    #----------------------------------------

    def getFullROCcurve(self, epoch, isTrain):
        # @return auc, numEvents, fpr, tpr
        #
        # epoch can also be 'BDT', otherwise a number

        # for the moment, just rerun the ROC curve calculation 
        # every time this is called
        # TODO: cache

        inputFname = self.__getInputFname(epoch, isTrain)

        auc, numEvents, fpr, tpr, thresholds = self.readROC(inputFname, isTrain, returnFullCurve = True)

        return auc, numEvents, fpr, tpr, thresholds

    #----------------------------------------

    def hasBDTroc(self, isTrain):
        if isTrain:
            return self.mvaROCfnames['train'] != None
        else:
            return self.mvaROCfnames['test'] != None

    #----------------------------------------

    def getInputDir(self):
        return self.resultDirData.inputDir

    #----------------------------------------

    def getInputDirDescription(self):
        return self.resultDirData.description

    #----------------------------------------

    def getAllROCs(self):
        # gets all non-exlucded roc values
        # calculates them if not in the cache

        # TODO: caching of the results

        mvaROC, rocValues = self.readROCfiles(ReadROChelper(self), 
                                         includeCached = True)
        return mvaROC, rocValues

#----------------------------------------------------------------------
