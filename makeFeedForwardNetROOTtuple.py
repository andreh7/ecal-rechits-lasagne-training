#!/usr/bin/env python

# makes a ROOT ntuple for one of the feed forward network trainings

import sys, os
import numpy as np

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]

assert len(ARGV) == 2, "usage: makeFeedForwardNetROOTtuple.py result-directory epoch"

outputDir, epoch = ARGV
epoch = int(epoch)

#----------------------------------------
sample = "test"
outputFname = "out.root"

#----------
# load input variables
#----------

import cPickle as pickle
inputData = pickle.load(open(os.path.join(outputDir, "input-%s.pkl" % sample)))

assert isinstance(inputData, list)
assert len(inputData) == 1

inputData = inputData[0]

numRows, numVars = inputData.shape

#----------
# load weights and labels
#----------
weightsLabelsFile = os.path.join(outputDir, "weights-labels-" + sample + ".npz")

weightsLabels = np.load(weightsLabelsFile)

if sample == 'train':
    weightVarName = "trainWeight"
else:
    # test sample
    weightVarName = "weight"

weights = weightsLabels[weightVarName]
labels  = weightsLabels['label']

assert len(labels) == numRows, "len(labels)=%d, numRows=%d" % (len(labels), numRows)
assert len(weights) == numRows

#----------
# load the NN output
#----------

outputData = np.load(os.path.join(outputDir, "roc-data-test-%04d.npz" % epoch))

output = outputData['output']

#----------
# write the root output tree
#----------
import ROOT
fout = ROOT.TFile(outputFname, "RECREATE")

tree = ROOT.TNtuple("tree", "tree",
                  
                  ":".join(
        # input variables
        [ "input%02d" % num for num in range(numVars) ] + 
        
        [ "weight", "label", "output" ]
        ))


import array
for row in range(numRows):
    
    values = list(inputData[row,:]) + [ weights[row], labels[row], output[row]]
                  
    tree.Fill(array.array('f', values))
    


tree.Write()
ROOT.gROOT.cd()
fout.Close()          

print >> sys.stderr,"wrote",outputFname

