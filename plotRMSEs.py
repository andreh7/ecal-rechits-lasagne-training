#!/usr/bin/env python

# given a results-* directory, plots the evolution
# of RMSE for regression tasks
# (similar to plotROCs.py for classification tasks)
# 
# no support for Torch output files so far

import glob, re, os, sys
import numpy as np

#----------------------------------------------------------------------

class ResultDirData:
    # keeps data which is common for the entire result directory
    def __init__(self, inputDir, useWeightsAfterPtEtaReweighting):
        self.inputDir = inputDir

        self.useWeightsAfterPtEtaReweighting = useWeightsAfterPtEtaReweighting

        self.description = readDescription(inputDir)

        # we don't have this for older trainings
        self.trainWeightsBeforePtEtaReweighting = None

        # check for dedicated weights and labels file
        # train dataset
        fname = os.path.join(inputDir, "weights-targets-train.npz")

        assert os.path.exists(fname)
        data = np.load(fname)

        # weights used during training (e.g. AFTER pt/eta reweighting)
        self.trainTrainWeights = data['weight']
        
        # weights to be used for plotting
        if 'plotWeights' in data.keys():
            self.trainPlotWeights = data['plotWeights']
        else:
            self.trainPlotWeights = None

        self.trainTargets = data['target']

        #----------
        # test dataset
        #----------

        fname = os.path.join(inputDir, "weights-targets-test.npz")
        assert os.path.exists(fname)
        data = np.load(fname)
        self.testWeights = data['weight']
        self.testTargets = data['target']

    #----------------------------------------

    def getWeights(self, isTrain):
        if isTrain:

            # for training, returns the weights before eta/pt reweighting if available
            # self.trainWeightsBeforePtEtaReweighting is None does not work,
            # 
            # if there are no trainWeightsBeforePtEtaReweighting, these are array(None, dtype=object)

            # prefer plotting weights (i.e. before reweighting)
            if self.trainPlotWeights != None:
                return self.trainPlotWeights
            else:
                return self.trainTrainWeights
        else:
            return self.testWeights

    #----------------------------------------

    def getTargets(self, isTrain):
        if isTrain:
            return self.trainTargets
        else:
            return self.testTargets

    #----------------------------------------
            
#----------------------------------------------------------------------

def addTimestamp(inputDir, x = 0.0, y = 1.07, ha = 'left', va = 'bottom'):

    import pylab, time


    # static variable
    if not hasattr(addTimestamp, 'text'):
        # make all timestamps the same during one invocation of this script

        now = time.time()

        addTimestamp.text = time.strftime("%a %d %b %Y %H:%M", time.localtime(now))

        # use the timestamp of the samples.txt file
        # as the starting point of the training
        # to determine the wall clock time elapsed
        # for the training

        fname = os.path.join(inputDir, "samples.txt")
        if os.path.exists(fname):
            startTime = os.path.getmtime(fname)
            deltaT = now - startTime

            addTimestamp.text += " (%.1f days)" % (deltaT / 86400.)


    pylab.gca().text(x, y, addTimestamp.text,
                     horizontalalignment = ha,
                     verticalalignment = va,
                     transform = pylab.gca().transAxes,
                     # color='green', 
                     fontsize = 10,
                     )
#----------------------------------------------------------------------

    
def addDirname(inputDir, x = 1.0, y = 1.07, ha = 'right', va = 'bottom'):

    import pylab

    if inputDir.endswith('/'):
        inputDir = inputDir[:-1]

    pylab.gca().text(x, y, inputDir,
                     horizontalalignment = ha,
                     verticalalignment = va,
                     transform = pylab.gca().transAxes,
                     # color='green', 
                     fontsize = 10,
                     )

#----------------------------------------------------------------------

def addNumEvents(numEventsTrain, numEventsTest):

    for numEvents, label, x0, halign in (
        (numEventsTrain, 'train', 0.00, 'left'),
        (numEventsTest, 'test',   1.00, 'right'),
        ):

        if numEvents != None:
            pylab.gca().text(x0, -0.08, '# ' + label + ' ev.: ' + str(numEvents),
                             horizontalalignment = halign,
                             verticalalignment = 'center',
                             transform = pylab.gca().transAxes,
                             fontsize = 10,
                             )

#----------------------------------------------------------------------

def readRMSE(resultDirData, fname, isTrain):
    # also looks for a cached file

    if fname.endswith(".cached-rmse.py"):

        # read the cached file
        print "reading",fname
        rmse = float(open(fname).read())
        return rmse

    print "reading",fname
    
    assert fname.endswith(".npz")
    data = np.load(fname)

    weights = resultDirData.getWeights(isTrain)
    targets  = resultDirData.getTargets(isTrain)
    outputs = data['output']

    from sklearn.metrics import mean_squared_error
    rmseValue = mean_squared_error(targets, outputs, sample_weight = weights)

    # write to cache
    cachedFname = fname + ".cached-rmse.py"
    fout = open(cachedFname,"w")
    print >> fout,rmseValue
    fout.close()

    # also copy the timestamp so that we can 
    # use it for estimating the time elapsed
    # for the plot
    modTime = os.path.getmtime(fname)
    os.utime(cachedFname, (modTime, modTime))

    return rmseValue

#----------------------------------------------------------------------

def readDescription(inputDir):
    descriptionFile = os.path.join(inputDir, "samples.txt")

    if os.path.exists(descriptionFile):

        description = []

        # assume that these are file names (of the training set)
        fnames = open(descriptionFile).read().splitlines()

        for fname in fnames:
            if not fname:
                continue

            fname = os.path.basename(fname)
            fname = os.path.splitext(fname)[0]

            if fname.endswith("-train"):
                fname = fname[:-6]
            elif fname.endswith("-test"):
                fname = fname[:-5]

            fname = fname.replace("_rechits","")

            description.append(fname)

        return ", ".join(description)

    else:
        return None

#----------------------------------------------------------------------

def readRMSEfiles(resultDirData, transformation = None, includeCached = False, maxEpoch = None,
                 excludedEpochs = None):
    # transformation is a function taking the file name 
    # which is run on each file
    # found and stored in the return values. If None,
    # just the name is stored.

    if transformation == None:
        transformation = lambda resultDirData, fname, isTrain: fname

    inputDir = resultDirData.inputDir

    #----------
    inputFiles = []

    if includeCached:
        # read cached version first
        inputFiles += glob.glob(os.path.join(inputDir, "rmse-data-*.npz.cached-rmse.py")) 

    inputFiles += glob.glob(os.path.join(inputDir, "rmse-data-*.npz")) 

    if not inputFiles:
        print >> sys.stderr,"no files rmse-data-* found, exiting"
        sys.exit(1)

    # RMSE values and epoch numbers for training and test
    # first index is 'train or 'test'
    # second index is epoch number
    rmseValues    = dict(train = {}, test = {})

    for inputFname in inputFiles:

        basename = os.path.basename(inputFname)

        # example names:
        #  rmse-data-train-0002.npz

        mo = re.match("rmse-data-(\S+)-(\d+)\.npz$", basename)

        if not mo and includeCached:
            mo = re.match("roc-data-(\S+)-(\d+)\.npz\.cached-rmse\.py$", basename)

        if mo:
            sampleType = mo.group(1)
            epoch = int(mo.group(2), 10)
            isTrain = sampleType == 'train'

            if maxEpoch == None or epoch <= maxEpoch:
                if excludedEpochs == None or not epoch in excludedEpochs:

                    if rmseValues[sampleType].has_key(epoch):
                        # skip reading this file: we already have a value
                        # (priority is given to the cached files)
                        continue

                rmseValues[sampleType][epoch] = transformation(resultDirData, inputFname, isTrain)
            continue

        print >> sys.stderr,"WARNING: unmatched filename",inputFname

    return rmseValues

#----------------------------------------------------------------------

def findLastCompleteEpoch(rmseFnames, ignoreTrain):
    
    trainEpochNumbers = sorted(rmseFnames['train'].keys())
    testEpochNumbers  = sorted(rmseFnames['test'].keys())

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

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser("""

      usage: %prog [options] result-directory

    """
    )

    parser.add_option("--ignore-train",
                      dest = 'ignoreTrain',
                      default = False,
                      action="store_true",
                      help="do not look at train values",
                      )

    parser.add_option("--max-epoch",
                      dest = 'maxEpoch',
                      type = int,
                      default = None,
                      help="last epoch to plot (useful e.g. if the training diverges at some point)",
                      )

    parser.add_option("--exclude-epochs",
                      dest = 'excludedEpochs',
                      type = str,
                      default = None,
                      help="comma separated list of epochs to ignore (e.g. in case of problematic/missing output files)",
                      )

    parser.add_option("--save-plots",
                      dest = 'savePlots',
                      default = False,
                      action="store_true",
                      help="save plots in input directory",
                      )

    parser.add_option("--legend-loc",
                      dest = 'legendLocation',
                      default = 'lower right',
                      help="location of legend in plots",
                      )

    # parser.add_option("--weights-after-pt-eta-reweighting",
    #                   dest = 'useWeightsAfterPtEtaReweighting',
    #                   default = False,
    #                   action = "store_true",
    #                   help="use weights (for training) after pt/eta reweighting",
    #                   )


    (options, ARGV) = parser.parse_args()

    assert len(ARGV) == 1, "usage: plotRMSEs.py result-directory"

    inputDir = ARGV.pop(0)

    if options.excludedEpochs != None:
        options.excludedEpochs = [ int(x) for x in options.excludedEpochs.split(',') ]

    #----------

    resultDirData = ResultDirData(inputDir, useWeightsAfterPtEtaReweighting = True)

    import pylab

    if True:
        #----------
        # plot evolution of RMSE vs. epoch
        #----------

        rmseValues = readRMSEfiles(resultDirData, 
                                   readRMSE, 
                                   includeCached = True, 
                                   maxEpoch = options.maxEpoch,
                                   excludedEpochs = options.excludedEpochs)

        print "plotting RMSE evolution"

        pylab.figure(facecolor='white')

        for sample, color in (
            ('train', 'blue'),
            ('test', 'red'),
            ):

            if options.ignoreTrain and sample == 'train':
                continue

            # sorted by ascending epoch
            epochs = sorted(rmseValues[sample].keys())
            rmses = [ rmseValues[sample][epoch] for epoch in epochs ]

            pylab.plot(epochs, rmses, '-o', label = "NN " + sample + " (last rmse=%.3f)" % rmses[-1], color = color, linewidth = 2)


        pylab.grid()
        pylab.xlabel('training epoch')
        pylab.ylabel('RMSE')

        pylab.legend(loc = options.legendLocation)

        if resultDirData.description != None:
            pylab.title(resultDirData.description)

        addTimestamp(inputDir)
        addDirname(inputDir)

        if options.savePlots:
            for suffix in (".png", ".pdf", ".svg"):
                outputFname = os.path.join(inputDir, "rmse-evolution" + suffix)
                pylab.savefig(outputFname)
                print "saved figure to",outputFname

    #----------

    if False:
        raise Exception("not yet implemented")

        #----------
        # plot correlation of train and test RMSE
        #----------

        import plotAUCcorr

        plotAUCcorr.doPlot(resultDirData, maxEpoch = options.maxEpoch,
                           excludedEpochs = options.excludedEpochs)

        if options.savePlots:
            for suffix in (".png", ".pdf", ".svg"):
                outputFname = os.path.join(inputDir, "auc-corr" + suffix)
                pylab.savefig(outputFname)
                print "saved figure to",outputFname


    #----------

    pylab.show()


