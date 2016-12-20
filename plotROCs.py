#!/usr/bin/env python

# given a results-* directory, finds the 
# .npz files with the network outputs
# for different epochs, calculates the ROC area
# and plots the progress

import sys, os

import glob, re
import numpy as np

#----------------------------------------------------------------------

class ResultDirData:
    # keeps data which is common for the entire result directory
    def __init__(self, inputDir, useWeightsAfterPtEtaReweighting):
        self.inputDir = inputDir

        self.useWeightsAfterPtEtaReweighting = useWeightsAfterPtEtaReweighting

        self.description = readDescription(inputDir)

        # we don't have this for older trainings
        # self.trainWeightsBeforePtEtaReweighting = None

        # check for dedicated weights and labels file
        # train dataset
        fname = os.path.join(inputDir, "weights-labels-train.npz")

        if os.path.exists(fname):
            data = np.load(fname)
            self.origTrainWeights = data['origTrainWeights']
            # self.trainWeightsBeforePtEtaReweighting = data['weightBeforePtEtaReweighting']
            self.trainLabels = data['label']
        else:
            # try the BDT file (but we don't have weights before eta/pt reweighting there)
            fname = os.path.join(inputDir, "roc-data-%s-mva.npz" % "train")
            data = np.load(fname)
            self.trainWeights = data['weight']
            self.trainLabels = data['label']
            self.trainWeightsBeforePtEtaReweighting = None
            
        #----------
        # test dataset
        #----------

        fname = os.path.join(inputDir, "weights-labels-test.npz")
        if os.path.exists(fname):
            data = np.load(fname)
            self.testWeights = data['weight']
            self.testLabels = data['label']
        else:
            # try the BDT file
            fname = os.path.join(inputDir, "roc-data-%s-mva.npz" % "test")
            data = np.load(fname)
            self.testWeights = data['weight']
            self.testLabels = data['label']

    #----------------------------------------

    def getWeights(self, isTrain):
        if isTrain:

            # for training, returns the weights before eta/pt reweighting if available
            # self.trainWeightsBeforePtEtaReweighting is None does not work,
            # 
            # if there are no trainWeightsBeforePtEtaReweighting, these are array(None, dtype=object)

            
            # if self.useWeightsAfterPtEtaReweighting:
            #     assert self.hasTrainWeightsBeforePtEtaReweighting()
            #     return self.trainWeights
            # 
            # if self.trainWeightsBeforePtEtaReweighting.shape == (): 
            #     return self.trainWeights
            # else:
            #     return self.trainWeightsBeforePtEtaReweighting

            # original weights, before any reweighting
            return self.origTrainWeights

        else:
            return self.testWeights

    #----------------------------------------

    def getLabels(self, isTrain):
        if isTrain:
            return self.trainLabels
        else:
            return self.testLabels

    #----------------------------------------
            
    def hasTrainWeightsBeforePtEtaReweighting(self):
        return self.trainWeightsBeforePtEtaReweighting.shape != ()


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

def readROC(resultDirData, fname, isTrain, returnFullCurve = False):
    # reads a torch file and calculates the area under the ROC
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
    
    assert fname.endswith(".npz")
    data = np.load(fname)

    weights = resultDirData.getWeights(isTrain)
    labels  = resultDirData.getLabels(isTrain)
    outputs = data['output']

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, dummy = roc_curve(labels, outputs, sample_weight = weights)

    aucValue = auc(fpr, tpr, reorder = True)

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

    if returnFullCurve:
        return aucValue, len(weights), fpr, tpr
    else:
        return aucValue

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

def readROCfiles(resultDirData, transformation = None, includeCached = False, maxEpoch = None,
                 excludedEpochs = None):
    # returns mvaROC, rocValues
    # which are dicts of 'test'/'train' to the single value
    # (for MVAid) or a dict epoch -> values (rocValues)
    #
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
        inputFiles += glob.glob(os.path.join(inputDir, "roc-data-*.npz.cached-auc.py")) 

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

    for inputFname in inputFiles:

        basename = os.path.basename(inputFname)

        # example names:
        #  roc-data-test-mva.npz
        #  roc-data-train-0002.npz

        mo = re.match("roc-data-(\S+)-mva\.npz$", basename)

        if mo:
            sampleType = mo.group(1)

            assert mvaROC.has_key(sampleType)
            assert mvaROC[sampleType] == None

            isTrain = sampleType == 'train'

            mvaROC[sampleType] = transformation(resultDirData, inputFname, isTrain)

            continue

        mo = re.match("roc-data-(\S+)-(\d+)\.npz$", basename)

        if not mo and includeCached:
            mo = re.match("roc-data-(\S+)-(\d+)\.npz\.cached-auc\.py$", basename)

        if mo:
            sampleType = mo.group(1)
            epoch = int(mo.group(2), 10)
            isTrain = sampleType == 'train'

            if maxEpoch == None or epoch <= maxEpoch:
                if excludedEpochs == None or not epoch in excludedEpochs:

                    if rocValues[sampleType].has_key(epoch):
                        # skip reading this file: we already have a value
                        # (priority is given to the cached files)
                        continue

                rocValues[sampleType][epoch] = transformation(resultDirData, inputFname, isTrain)
            continue

        print >> sys.stderr,"WARNING: unmatched filename",inputFname

    return mvaROC, rocValues

#----------------------------------------------------------------------

def drawSingleROCcurve(resultDirData, inputFname, isTrain, label, color, lineStyle, linewidth):

    auc, numEvents, fpr, tpr = readROC(resultDirData, inputFname, isTrain, returnFullCurve = True)

    # TODO: we could add the area to the legend
    pylab.plot(fpr, tpr, lineStyle, color = color, linewidth = linewidth, label = label.format(auc = auc))

    return fpr, tpr, numEvents

#----------------------------------------------------------------------

def plotGradientMagnitudes(inputDir, mode):

    # @param mode can be 
    #    'stat'   plot the median/mean and error bars
    #    'detail' plot the actual gradient magnitude values
    #
    # @return True if something was plotted

    inputFiles = glob.glob(os.path.join(inputDir, "gradient-magnitudes-*.npz"))
    
    # maps from epoch number to gradient magnitudes
    epochToGradientMagnitudes = {}

    # sort by epoch number
    for inputFname in inputFiles:

        basename = os.path.basename(inputFname)

        # example names:
        #  roc-data-test-mva.t7
        #  roc-data-train-0002.t7

        mo = re.match("gradient-magnitudes-(\d+).npz$", basename)
        if not mo:
            print >> sys.stderr,"warning: skipping",inputFname
            continue
        epoch = int(mo.group(1))
        
        thisData = np.load(inputFname)['gradientMagnitudes']

        assert not epoch in epochToGradientMagnitudes

        epochToGradientMagnitudes[epoch] = thisData

    # end of loop over input files

    if not epochToGradientMagnitudes:
        return False

    pylab.figure(facecolor='white')

    # median number of gradient evaluations per epoch
    # (for filling in missing ones)
    # normally they should all be the same
    medianLength = np.median(
        [ len(x) for x in epochToGradientMagnitudes.values() ]
        )

    epochs = sorted(epochToGradientMagnitudes.keys())


    if mode == 'detail':
        # plot epoch by epoch with different colors
        for epoch in epochs:
            yvalues = epochToGradientMagnitudes[epoch]

            # choose x axis normalization such that each epoch
            # corresponds to an interval of one
            xvalues = np.linspace(epoch, epoch + 1,
                                  num = len(yvalues),
                                  endpoint = False)

            pylab.plot(xvalues, yvalues)

    elif mode == 'stat':
        # plot mean and standard deviations
        xvalues = epochs

        yvalues = [ np.mean(epochToGradientMagnitudes[epoch]) for epoch in epochs ]
        yerrs   = [ np.std(epochToGradientMagnitudes[epoch]) for epoch in epochs ]

        pylab.errorbar(xvalues, yvalues, yerr = yerrs)

    else:
        raise Exception("unsupported mode " + mode)

    pylab.xlabel('epoch')
    pylab.ylabel('gradient magnitude')

    pylab.grid()

    # minor ticks to indicate start of epochs
    ax = pylab.gca()
    ax.set_xticks(epochs, minor = True)
    ax.grid(which='minor', alpha=0.2)                                                




    return True

#----------------------------------------------------------------------

def findLastCompleteEpoch(rocFnames, ignoreTrain):
    
    trainEpochNumbers = sorted(rocFnames['train'].keys())
    testEpochNumbers  = sorted(rocFnames['test'].keys())

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
def updateHighestTPR(highestTPRs, fpr, tpr, maxfpr):
    if maxfpr == None:
        return

    # find highest TPR for which the FPR is <= maxfpr
    highestTPR = max([ thisTPR for thisTPR, thisFPR in zip(tpr, fpr) if thisFPR <= maxfpr])
    highestTPRs.append(highestTPR)

#----------------------------------------------------------------------
def drawLast(inputDirData, xmax = None, ignoreTrain = False, maxEpoch = None, 
             excludedEpochs = None,
             savePlots = False,
             legendLocation = None
             ):
    # plot ROC curve for last epoch only
    pylab.figure(facecolor='white')
    
    # read only the file names
    mvaROC, rocFnames = readROCfiles(inputDirData, maxEpoch = maxEpoch, 
                                     excludedEpochs = excludedEpochs)

    #----------

    # find the highest epoch for which both
    # train and test samples are available

    epochNumber = findLastCompleteEpoch(rocFnames, ignoreTrain)

    highestTPRs = []
    #----------

    # maps from sample type to number of events
    numEvents = {}

    for sample, color in (
        ('train', 'blue'),
        ('test', 'red'),
        ):

        isTrain = sample == 'train'

        if ignoreTrain and isTrain:
            continue
        
        # take the last epoch
        if epochNumber != None:
            fpr, tpr, numEvents[sample] = drawSingleROCcurve(inputDirData, rocFnames[sample][epochNumber], isTrain, "NN " + sample + " (auc {auc:.3f})", color, '-', 2)
            updateHighestTPR(highestTPRs, fpr, tpr, xmax)
            

        # draw the ROC curve for the MVA id if available
        fname = mvaROC[sample]
        if fname != None:
            fpr, tpr, dummy = drawSingleROCcurve(inputDirData, fname, isTrain, "BDT " + sample + " (auc {auc:.3f})", color, '--', 1)
            updateHighestTPR(highestTPRs, fpr, tpr, xmax)            

    pylab.xlabel('fraction of false positives')
    pylab.ylabel('fraction of true positives')

    if xmax != None:
        pylab.xlim(xmax = xmax)
        # adjust y scale
        pylab.ylim(ymax = 1.1 * max(highestTPRs))

    pylab.grid()
    pylab.legend(loc = legendLocation)

    addTimestamp(inputDirData.inputDir)
    addDirname(inputDirData.inputDir)
    addNumEvents(numEvents.get('train', None), numEvents.get('test', None))

    if inputDirData.description != None:

        title = str(inputDirData.description)

        if epochNumber != None:
            title += " (epoch %d)" % epochNumber

        pylab.title(title)

    if savePlots:
        for suffix in (".png", ".pdf", ".svg"):
            outputFname = os.path.join(inputDirData.inputDir, "last-auc")

            if xmax != None:
                outputFname += "-%.2f" % xmax

            outputFname += suffix

            pylab.savefig(outputFname)
            print "saved figure to",outputFname

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser("""

      usage: %prog [options] result-directory

    """
    )

    parser.add_option("--last",
                      default = False,
                      action="store_true",
                      help="plot ROC curve for last epoch only",
                      )

    parser.add_option("--both",
                      default = False,
                      action="store_true",
                      help="plot AUC evolution and last AUC curve",
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

    parser.add_option("--weights-after-pt-eta-reweighting",
                      dest = 'useWeightsAfterPtEtaReweighting',
                      default = False,
                      action = "store_true",
                      help="use weights (for training) after pt/eta reweighting",
                      )


    (options, ARGV) = parser.parse_args()

    assert len(ARGV) == 1, "usage: plotROCs.py result-directory"

    inputDir = ARGV.pop(0)

    if options.excludedEpochs != None:
        options.excludedEpochs = [ int(x) for x in options.excludedEpochs.split(',') ]

    #----------

    resultDirData = ResultDirData(inputDir, options.useWeightsAfterPtEtaReweighting)

    import pylab

    if options.last or options.both:

        drawLast(resultDirData, ignoreTrain = options.ignoreTrain, maxEpoch = options.maxEpoch, 
                 excludedEpochs = options.excludedEpochs,
                 savePlots = options.savePlots,
                 legendLocation = options.legendLocation)

        # zoomed version
        # autoscaling in y with x axis range manually
        # set seems not to work, so we implement
        # something ourselves..
        drawLast(resultDirData, xmax = 0.05, ignoreTrain = options.ignoreTrain, maxEpoch = options.maxEpoch, 
                 excludedEpochs = options.excludedEpochs,
                 savePlots = options.savePlots,
                 legendLocation = options.legendLocation
                 )


    if not options.last or options.both:
        #----------
        # plot evolution of area under ROC curve vs. epoch
        #----------

        mvaROC, rocValues = readROCfiles(resultDirData, 
                                         readROC, 
                                         includeCached = True, 
                                         maxEpoch = options.maxEpoch,
                                         excludedEpochs = options.excludedEpochs)

        print "plotting AUC evolution"

        pylab.figure(facecolor='white')

        for sample, color in (
            ('train', 'blue'),
            ('test', 'red'),
            ):

            if options.ignoreTrain and sample == 'train':
                continue

            # sorted by ascending epoch
            epochs = sorted(rocValues[sample].keys())
            aucs = [ rocValues[sample][epoch] for epoch in epochs ]

            pylab.plot(epochs, aucs, '-o', label = "NN " + sample + " (last auc=%.3f)" % aucs[-1], color = color, linewidth = 2)

            # draw a line for the MVA id ROC if available
            auc = mvaROC[sample]
            if auc != None:
                pylab.plot( pylab.gca().get_xlim(), [ auc, auc ], '--', color = color, 
                            label = "BDT %s (auc=%.3f)" % (sample, auc))

        pylab.grid()
        pylab.xlabel('training epoch')
        pylab.ylabel('AUC')

        pylab.legend(loc = options.legendLocation)

        if resultDirData.description != None:
            pylab.title(resultDirData.description)

        addTimestamp(inputDir)
        addDirname(inputDir)

        if options.savePlots:
            for suffix in (".png", ".pdf", ".svg"):
                outputFname = os.path.join(inputDir, "auc-evolution" + suffix)
                pylab.savefig(outputFname)
                print "saved figure to",outputFname

    #----------

    if not options.last or options.both:

        #----------
        # plot correlation of train and test AUC
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
        # plot gradient magnitudes
        #----------

        plotted = plotGradientMagnitudes(inputDir, mode = 'detail')

        if plotted and options.savePlots:
            for suffix in (".png", ".pdf", ".svg"):
                outputFname = os.path.join(inputDir, "gradient-magnitude" + suffix)
                pylab.savefig(outputFname)
                print "saved figure to",outputFname


    #----------

    if not options.savePlots:
        # show plots interactively
        pylab.show()


