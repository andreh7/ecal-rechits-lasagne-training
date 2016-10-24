#!/usr/bin/env python

# from sgdWithLearningRateDecay(..) in train01.py

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

learningRate = LearningRate(learningRate = 1e-3,
                  learningRateDecay = 1e-7)

numEpochs = 324

batchSize = 32
numTrainSamples = 3424581

# round down
iterationsPerEpoch = numTrainSamples / batchSize

yvalues = []
xvalues = []

iternum = 0
for epoch in range(1, numEpochs + 1):

    xvalues.extend(np.linspace(epoch, epoch + 1,
                               iterationsPerEpoch,
                               endpoint = False))
    
    for i in range(iterationsPerEpoch):
        yvalues.append(learningRate(iternum))
        iternum += 1

import pylab
pylab.figure(facecolor='white')

pylab.plot(xvalues, yvalues)
pylab.grid()
pylab.xlabel('epoch number')
pylab.ylabel('learning rate')
pylab.savefig("/tmp/learning-rate-evolution.pdf")
print "plotting done"
pylab.show()
