#!/usr/bin/env python

# plot weight evolution vs. training epoch
# for each group of weights separately

import sys, os, glob, re, pylab, math
import numpy as np

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

if __name__ == '__main__':
    ARGV = sys.argv[1:]

    assert len(ARGV) == 1

    inputDir = ARGV.pop(0)
    
    # load the data
    inputFiles = glob.glob(os.path.join(inputDir,"model-*.npz"))

    # first index is object (within the model) name
    # second index is epoch number,
    modelData = {}

    allEpochs = []

    for inputFname in inputFiles:
        
        basename = os.path.basename(inputFname)
        mo = re.match("model-(\d+).npz$", basename)

        if not mo:
            print >> sys.stderr,"WARNING: unmatched filename",inputfname
            continue

        epoch = int(mo.group(1), 10)

        allEpochs.append(epoch)

        data = np.load(inputFname)

        # assume there is no hierarchy, just one layer
        for key in data.keys():
            modelData.setdefault(key, {})[epoch] = data[key]

    # end of loop over input files

    allEpochs.sort()

    #----------
    # plot
    #----------

    # quantiles to be plotted

    cdf = lambda z: 0.5 * (1+math.erf(z / math.sqrt(2)))

    oneSigma = [ cdf(-1), cdf(+1) ]

    quantiles2 = [ 0.05, 0.95 ]

    for key in sorted(modelData.keys()):
        
        # for each epoch, get summarizing statistics such as 
        # min/max, average, standard deviation etc.

        pylab.figure(facecolor='white')

        thisEpochs = sorted(modelData[key].keys())

        yvalues = [ np.median(np.abs(modelData[key][epoch])) for epoch in thisEpochs ]
        
        yerrs = np.zeros((2, len(thisEpochs)))

        for index in range(len(thisEpochs)):
            epoch = thisEpochs[index]
            yerrs[0,index] = np.percentile(np.abs(modelData[key][epoch]), 100 * oneSigma[0])
            yerrs[1,index] = np.percentile(np.abs(modelData[key][epoch]), 100 * oneSigma[1]) 

        assert yerrs.min() >= 0

        # both must be positive and relative to yvalues
        yerrs[0] = yvalues - yerrs[0]
        yerrs[1] = yerrs[1] - yvalues

        pylab.errorbar(thisEpochs,
                       yvalues,
                       yerrs)


        #----------
        # 5/95% quantiles
        #----------

        yerrs = np.zeros((2, len(thisEpochs)))

        for index in range(len(thisEpochs)):
            epoch = thisEpochs[index]
            yerrs[0,index] = np.percentile(np.abs(modelData[key][epoch]), 100 * quantiles2[0])
            yerrs[1,index] = np.percentile(np.abs(modelData[key][epoch]), 100 * quantiles2[1]) 

        assert yerrs.min() >= 0

        pylab.plot(thisEpochs, yerrs[0,:], '*', c = 'orange')
        pylab.plot(thisEpochs, yerrs[1,:], '*', c = 'orange')

        #----------
        # plot maximum and minimum
        #----------
        ymax = [ np.max(np.abs(modelData[key][epoch]))  for epoch in thisEpochs ]
        ymin = [ np.min(np.abs(modelData[key][epoch]))  for epoch in thisEpochs ]

        pylab.plot(thisEpochs, ymax, '*', c = 'red')
        pylab.plot(thisEpochs, ymin, '*', c = 'red')

        pylab.xlabel("epoch")
        pylab.ylabel("coefficient magnitudes")
        pylab.title(key + " (shape %s)" % str(
                modelData[key][thisEpochs[0]].shape
                )
                    )

        pylab.grid()

        #----------
        # save plot
        #----------
        for suffix in ("png","pdf"):
            pylab.savefig(os.path.join(inputDir,
                                       "weight-summary-" + key + "." + suffix))

    # end of loop over plots

     


    # pylab.show()
