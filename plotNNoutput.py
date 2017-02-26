#!/usr/bin/env python

import sys, os
import numpy as np

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

from optparse import OptionParser
parser = OptionParser("""

  usage: %prog [options] result-directory epoch

"""
)

parser.add_option("--save-plots",
                  dest = 'savePlots',
                  default = False,
                  action="store_true",
                  help="save plots in input directory",
                  )

parser.add_option("--sample",
                  dest = 'sample',
                  default = "test",
                  choices = [ "test", "train" ],
                  help="sample to use (train or test)",
                  )


(options, ARGV) = parser.parse_args()
assert len(ARGV) == 2, "usage: plotNNoutput.py result-directory epoch"

outputDir, epoch = ARGV
epoch = int(epoch)

#----------------------------------------


options.sample = "train"


weightsLabelsFile = os.path.join(outputDir, "weights-labels-" + options.sample + ".npz")

weightsLabels = np.load(weightsLabelsFile)

weights = weightsLabels[options.sample + 'Weight']
labels  = weightsLabels['label']

outputsFile = os.path.join(outputDir, "roc-data-%s-%04d.npz" % (options.sample, epoch))
outputsData = np.load(outputsFile)

output = outputsData['output']


import pylab
pylab.hist(output[labels == 1], weights = weights[labels == 1], bins = 100, label='signal', histtype = 'step')
pylab.hist(output[labels == 0], weights = weights[labels == 0], bins = 100, label='background', histtype = 'step')
pylab.legend()
pylab.xlabel('NN output')
pylab.title(options.sample + " epoch %d" % epoch)
pylab.grid()

if options.savePlots:
    outputFname = os.path.join(outputDir, "nn-output-" + options.sample + "-%04d.pdf" % epoch)
    pylab.savefig(outputFname)
    print >> sys.stderr,"wrote plots to",outputFname
else:
    pylab.show()
