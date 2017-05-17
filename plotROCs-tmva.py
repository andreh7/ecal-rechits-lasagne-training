#!/usr/bin/env python

# given a results-* directory with TMVA training,
# produces ROC curves for training and test samples
import sys, os

import glob, re
import numpy as np

from plotROCutils import addTimestamp, addDirname, addNumEvents, readDescription

officialPhotonIdLabel = 'official photon id'

#----------------------------------------------------------------------

def readROC(fname, isTrain, returnFullCurve = False, origMVA = False):
    # reads the TMVA output file under the ROC curve for it
    # 
    # @param origMVA if True, returns the values corresponding to the original 

    print "reading",fname
    
    assert fname.endswith(".root")

    if isTrain:
        treeName = 'TrainTree'
    else:
        treeName = 'TestTree'

    import ROOT

    fin = ROOT.TFile(fname)
    assert fin.IsOpen(), "could not open file " + fname

    tree = fin.Get(treeName)
    ROOT.gROOT.cd()

    assert tree != None, "could not find tree " + treeName

    if tree.GetEntries() > 1000000:
        tree.SetEstimate(tree.GetEntries())

    labels = []
    predictions = []
    weights = []

    if origMVA:
        # official photon id
        varname = "origmva"
    else:
        # our own training
        varname = "BDT"

    for className, label in (('Signal', 1), 
                             ('Background',0)):

        # note that class e.g. can be zero for signal etc.
        # so we better rely on className 
        tree.Draw(varname + ":origTrainWeights",'className == "%s"' % className,"goff")
        nentries = tree.GetSelectedRows()

        v1 = tree.GetV1(); v2 = tree.GetV2()

        labels      += [ label ] * nentries
        predictions += [ v1[i] for i in range(nentries) ]
        weights     += [ v2[i] for i in range(nentries) ]

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, dummy = roc_curve(labels, predictions, sample_weight = weights)

    aucValue = auc(fpr, tpr, reorder = True)

    ROOT.gROOT.cd()
    fin.Close()

    if returnFullCurve:
        return aucValue, fpr, tpr, len(weights)
    else:
        return aucValue

#----------------------------------------------------------------------

def drawSingleROCcurve(auc, fpr, tpr, label, color, lineStyle, linewidth):

    # TODO: we could add the area to the legend
    pylab.plot(fpr, tpr, lineStyle, color = color, linewidth = linewidth, label = label.format(auc = auc))


#----------------------------------------------------------------------
def updateHighestTPR(highestTPRs, fpr, tpr, maxfpr):
    if maxfpr == None:
        return

    # find highest TPR for which the FPR is <= maxfpr
    highestTPR = max([ thisTPR for thisTPR, thisFPR in zip(tpr, fpr) if thisFPR <= maxfpr])
    highestTPRs.append(highestTPR)

#----------------------------------------------------------------------
def drawROCcurves(tmvaOutputFname, xmax = None, ignoreTrain = False,  
             savePlots = False,
             legendLocation = None
             ):
    # plot ROC curve 

    inputDir = os.path.dirname(tmvaOutputFname)

    pylab.figure(facecolor='white')

    # first index is 'MVA' (for official photon id) or 'TMVA'
    # for our own training
    
    auc = {}
    fpr = {}
    tpr = {}
    numEvents = {}

    for key in (officialPhotonIdLabel, 'our training'):
        auc[key] = {}
        fpr[key] = {}
        tpr[key] = {}
        numEvents[key] = {}
        origMVA = key == officialPhotonIdLabel
        for sample in ('train', 'test'):
            auc[key][sample], fpr[key][sample], tpr[key][sample], numEvents[key][sample] = readROC(tmvaOutputFname, sample == 'train', True,
                                                                                                   origMVA = origMVA)

    #----------

    highestTPRs = []
    #----------

    # maps from sample type to number of events
    numEvents = {}

    for sample, color in (
        ('train', 'blue'),
        ('test', 'red'),
        ):

        for key, lineStyle, lineWidth in (
            ('our training', '-', 2),
            (officialPhotonIdLabel,  '--', 1),
            ):

            isTrain = sample == 'train'

            if ignoreTrain and isTrain:
                continue

            drawSingleROCcurve(auc[key][sample], fpr[key][sample], tpr[key][sample], 
                               key + " (" + sample + " auc {auc:.3f})", color, lineStyle, lineWidth)
            updateHighestTPR(highestTPRs, fpr[key][sample], tpr[key][sample], xmax)

    pylab.xlabel('fraction of false positives')
    pylab.ylabel('fraction of true positives')

    if xmax != None:
        pylab.xlim(xmax = xmax)
        # adjust y scale
        pylab.ylim(ymax = 1.1 * max(highestTPRs))

    pylab.grid()
    pylab.legend(loc = legendLocation)

    addTimestamp(inputDir)
    addDirname(inputDir)
    addNumEvents(numEvents.get('train', None), numEvents.get('test', None))

    description = readDescription(inputDir)

    if description != None:

        title = str(description)

        pylab.title(title)

    if savePlots:
        for suffix in (".png", ".pdf", ".svg"):
            outputFname = os.path.join(inputDir, "last-auc")

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

    parser.add_option("--ignore-train",
                      dest = 'ignoreTrain',
                      default = False,
                      action="store_true",
                      help="do not look at train values",
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

    (options, ARGV) = parser.parse_args()

    assert len(ARGV) == 1, "usage: plotROCs-tmva.py result-directory"

    inputDir = ARGV.pop(0)

    #----------

    tmvaOutputFname = os.path.join(inputDir,"tmva.root")
    if not os.path.exists(tmvaOutputFname):
        print >> sys.stderr,"file",tmvaOutputFname,"does not exist"
        sys.exit(1)

    import pylab

    drawROCcurves(tmvaOutputFname, ignoreTrain = options.ignoreTrain,
                  savePlots = options.savePlots,
                  legendLocation = options.legendLocation)

    # zoomed version
    # autoscaling in y with x axis range manually
    # set seems not to work, so we implement
    # something ourselves..
    drawROCcurves(tmvaOutputFname, xmax = 0.05, ignoreTrain = options.ignoreTrain,  
                  savePlots = options.savePlots,
                  legendLocation = options.legendLocation
                  )

    if not options.savePlots:
        pylab.show()


