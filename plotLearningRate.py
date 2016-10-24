#!/usr/bin/env python

# from sgdWithLearningRateDecay(..) in train01.py
import glob, os, re

import numpy as np
#----------------------------------------------------------------------

class LearningRate:
    def __init__(self, learningRate, learningRateDecay):
        self.learningRate = learningRate
        self.learningRateDecay = learningRateDecay


    def __call__(self, t):

        clr = self.learningRate / (1 + t * self.learningRateDecay)

        return clr
#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser(prog='plotLearningRate.py',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                 )

parser.add_argument('resultDir',
                    nargs = 1,
                    default = None,
                    metavar = "D",
                    help='result directory to infer parameters from',
                    )

options = parser.parse_args()

#----------------------------------------
if options.resultDir == None:
    print >> sys.stderr,"should specify the result directory"
    sys.exit(1)

# find number of epochs
fnames = glob.glob(os.path.join(options.resultDir[0],"roc-data-train-*.npz"))

if not fnames:
    print >> sys.stderr,"no roc-data-train-*.npz files found in",options.resultDir
    sys.exit(1)

# find latest epoch
maxEpoch = None
maxEpochFile = None
for fname in fnames:
    mo = re.search("roc-data-train-(\d+).npz$", fname)
    # 'MVA' does not match
    if not mo:
        continue

    thisEpoch = int(mo.group(1),10)
    if maxEpoch == None or thisEpoch > maxEpoch:
        maxEpoch = thisEpoch
        maxEpochFile = fname

    
assert maxEpoch != None

# find number of training samples 
data = np.load(maxEpochFile)
numTrainSamples = len(data['label'])    

#----------------------------------------

learningRate = LearningRate(learningRate = 1e-3,
                  learningRateDecay = 1e-7)

learningRate = np.frompyfunc(learningRate, 1, 1)

batchSize = 32

# round down
iterationsPerEpoch = numTrainSamples / batchSize

yvalues = []
xvalues = []

# we can't plot every iteration for large datasets
# and epoch numbers
# plot e.g. 1000 points 

numPoints = 1001

itervalues = np.linspace(0, maxEpoch * iterationsPerEpoch, numPoints)
xvalues    = np.linspace(1, maxEpoch, numPoints)

assert len(itervalues) == len(xvalues)
yvalues = learningRate(itervalues)

import pylab
pylab.figure(facecolor='white')

pylab.plot(xvalues, yvalues)
pylab.grid()
pylab.xlabel('epoch number')
pylab.ylabel('learning rate')

outputFname = os.path.join(options.resultDir[0], "learning-rate-evolution.pdf")
pylab.savefig(outputFname)
print "saved figure to",outputFname
pylab.show()
