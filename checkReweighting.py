#!/usr/bin/env python


import sys
ARGV = sys.argv[1:]

assert len(ARGV) == 1

fname = ARGV.pop(0)

import numpy as np
import pylab

data = np.load(fname)

weights = data['weights']
labels = data['labels']
pt = data['pt']

eta = data['eta']

ptBins = np.linspace(0,120, 30 + 1)
# assume barrel for the moment
etaBins = np.linspace(0, 1.6, 32 + 1)

if False:
    # normalize signal normalization to background normalization
    sumBg = sum(weights[labels == 0])
    sumSig = sum(weights[labels == 1])

    weights[labels == 1] *= sumBg / sumSig


for label, values, bins in (
    ('pt', pt, ptBins),
    ('eta', eta, etaBins)):

    pylab.figure(facecolor='white')

    contentsBg, b, patches = pylab.hist(values[labels == 0], bins = bins, weights = weights[labels == 0],
                                        histtype = 'step',
                                        label = 'non-prompt/fake',
                                        color = 'red')
    contentsSig, b, patches = pylab.hist(values[labels == 1], bins = bins, weights = weights[labels == 1],
                                            histtype = 'step',
                                            label = 'prompt',
                                            color = 'green')
    
    pylab.gca().set_ylim([
            pylab.gca().get_ylim()[0],
            max(list(contentsSig) + list(contentsBg)) * 1.1])
            

    pylab.legend()
    pylab.xlabel(label)
    pylab.grid()

    

pylab.show()
    
