#!/usr/bin/env python

import time
import numpy as np
import os, sys

from sklearn.metrics import roc_auc_score

from Timer import Timer

sys.path.append(os.path.expanduser("~/torchio")); import torchio

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

# parse command line arguments
import argparse

parser = argparse.ArgumentParser(prog='train-tmva.py',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                 )

parser.add_argument('--max-epochs',
                    dest = "maxEpochs",
                    default = None,
                    type = int,
                    help='stop after the given number of epochs',
                    )

parser.add_argument('--output-dir',
                    dest = "outputDir",
                    default = None,
                    help='manually specify the output directory',
                    )

parser.add_argument('--param',
                    dest = "params",
                    default = [],
                    help='additional python to be evaluated after reading model and dataset file. Can be used to change some parameters. Option can be specified multiple times.',
                    action = 'append',
                    )


parser.add_argument('modelFile',
                    metavar = "modelFile.py",
                    type = str,
                    nargs = 1,
                    help='file with model building code',
                    )

parser.add_argument('dataFile',
                    metavar = "dataFile.py",
                    type = str,
                    nargs = 1,
                    help='file with data loading code',
                    )

options = parser.parse_args()

#----------

execfile(options.modelFile[0])
execfile(options.dataFile[0])

for param in options.params:
    # python 2
    exec param

#----------
print "loading data"

doPtEtaReweighting = globals().get("doPtEtaReweighting", False)

cuda = True
with Timer("loading training dataset...") as t:
    trainData, trsize = datasetLoadFunction(dataDesc['train_files'], dataDesc['trsize'], 
                                            cuda = cuda, 
                                            isTraining = True,
                                            reweightPtEta = doPtEtaReweighting)
with Timer("loading test dataset...") as t:
    testData,  tesize = datasetLoadFunction(dataDesc['test_files'], dataDesc['tesize'], cuda, 
                                            isTraining = False,
                                            reweightPtEta = False)

# convert labels from -1..+1 to 0..1 for cross-entropy loss
# must clone to assign

# TODO: normalize these to same weight for positive and negative samples
trainWeights = trainData['weights'].reshape((-1,1))
testWeights  = testData['weights'].reshape((-1,1))

if doPtEtaReweighting:
    origTrainWeights = trainData['weightsBeforePtEtaReweighting']
else:
    # they're the same
    origTrainWeights = trainWeights

#----------
if options.outputDir == None:
    options.outputDir = "results/" + time.strftime("%Y-%m-%d-%H%M%S")

if not os.path.exists(options.outputDir):
    os.makedirs(options.outputDir)

# try to set the process name
try:
    import procname
    procname.setprocname("train " + 
                         os.path.basename(options.outputDir.rstrip('/')))
except ImportError, ex:
    pass

#----------
# write training file paths to result directory
#----------

fout = open(os.path.join(options.outputDir, "samples.txt"), "w")
for fname in dataDesc['train_files']:
    print >> fout, fname
fout.close()

#----------

logfile = open(os.path.join(options.outputDir, "train.log"), "w")

fouts = [ sys.stdout, logfile ]


#----------

for fout in fouts:
    print >> fout, "doPtEtaReweighting=",doPtEtaReweighting

#----------
# write out BDT/MVA id labels (for performance comparison)
#----------
for name, output in (
    ('train', trainData['mvaid']),
    ('test',  testData['mvaid']),
    ):
    np.savez(os.path.join(options.outputDir, "roc-data-%s-mva.npz" % name),
             # these are the BDT outputs
             output = output,
             )

# save weights (just once, we assume the ordering of the events is always the same)
np.savez(os.path.join(options.outputDir, "weights-labels-train.npz"),
         trainWeight = trainWeights,             
         origTrainWeights = origTrainWeights,
         doPtEtaReweighting = doPtEtaReweighting,
         label = trainData['labels'],
         )
np.savez(os.path.join(options.outputDir, "weights-labels-test.npz"),
         weight = testWeights,             
         label = testData['labels'],
         )

#----------
# convert targets to integers (needed for softmax)
#----------

for data in (trainData, testData):
    data['labels'] = data['labels'].astype('int32').reshape((-1,1))

#----------
# produce test and training input once
#----------
# assuming we have enough memory 
#
# TODO: can we use slicing instead of unpacking these again for the minibatches ?
with Timer("unpacking training dataset...", fouts) as t:
    trainInput = makeInput(trainData, range(len(trainData['labels'])), inputDataIsSparse = True)

    assert len(trainInput) == 1, "need exactly one train input set per event for TMVA"

    trainInput = trainInput[0]

    # (rows, columns)
    numInputVars = trainInput.shape[1]

with Timer("unpacking test dataset...", fouts) as t:
    testInput  = makeInput(testData, range(len(testData['labels'])), inputDataIsSparse = True)

    assert len(testInput) == 1, "need exactly one test input set per event for TMVA"
    testInput = testInput[0]

#----------
# initialize TMVA factory
#----------

import ROOT
ROOT.TMVA.Tools.Instance()
tmvaOutputFile = ROOT.TFile(os.path.join(options.outputDir, "tmva.root"),"RECREATE")

factory = ROOT.TMVA.Factory("TMVAClassification", tmvaOutputFile,
                            ":".join([
                                "!V",
                                "!Silent",
                                "Color",
                                "DrawProgressBar",
                                # "Transformations=I;D;P;G,D",
                                "AnalysisType=Classification"]
                                     ))

for varIndex in range(numInputVars):
    factory.AddVariable("var%02d" % varIndex,"F")

# needed because we use Add{Training,Test}Event(..) methods
factory.CreateEventAssignTrees("inputTree")
 
#----------
# convert input data for TMVA
#----------

# we add an additional variable to identify the training/test sample
with Timer("passing train+test data to TMVA factory...", fouts) as t:
    # varnames = [ "var%02d" % varIndex for varIndex in range(numInputVars) ] + [ "isTrain" ]
    
    # sigTuple = ROOT.TNtuple("sigTuple", "sigTuple", varnames.join(":"))
    # bgTuple  = ROOT.TNtuple("bgTuple",  "bgTuple",  varnames.join(":"))

    values = ROOT.vector('double')(numInputVars)

    for inputData, labels, weights, method in ((trainInput, trainData['labels'], trainData['weights'], factory.AddTrainingEvent),
                                               (testInput,  testData['labels'],  testData['weights'], factory.AddTestEvent)):

        for row in range(inputData.shape[0]):

            for index, value in enumerate(inputData[row]):
                values[index] = value
            
            if labels[row] == 1:
                method("Signal", values, weights[row])
            else:
                method("Background", values, weights[row])

factory.PrepareTrainingAndTestTree(ROOT.TCut(""),
                                   ",".join([
            "nTrain_Signal=%d" % len(testData['labels']),
            "nTrain_Background=%d" % len(testData['labels']),
            "SplitMode=Block",
            ]))

#----------
# run training with TMVA
#----------

method = factory.BookMethod(ROOT.TMVA.Types.kBDT, "BDT",
                   ":".join([

            # from https://raw.githubusercontent.com/InnaKucher/flashgg/0726271781a6a9379471cc1e848075cd6102db43/MicroAOD/data/MVAweights_80X_barrel_ICHEP_wShift.xml
                       "!H",
                       "!V",
                       "IgnoreNegWeightsInTraining",
                       "NTrees=1000",
                       "MaxDepth=3",
                       "nCuts=2000",
                       "BoostType=Grad",
                       "Shrinkage=1.000000e-01",
                       "!UseYesNoLeaf",
                       "!UseBaggedGrad",
                       ]))

print 'starting training at', time.asctime()
 
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

print "done with training at", time.asctime()

#----------
# evaluate AUC
#----------
ROOT.gROOT.cd()
tmvaOutputFile.Close()

# reopen
tmvaOutputFile = ROOT.TFile(tmvaFname)

for name, treeName in (('train', 'TrainTree'),
                       ('test', 'TestTree')):
    
    tree = tmvaOutputFile.Get(treeName)
    ROOT.gROOT.cd()

    assert tree != None, "could not find tree " + treeName

    if tree.GetEntries() > 1000000:
        tree.SetEstimate(tree.GetEntries())

    tree.Draw("classID:BDT:weight","","goff")
    nentries = tree.GetSelectedRows()

    v1 = tree.GetV1(); v2 = tree.GetV2(); v3 = tree.GetV3()

    labels      = [ v1[i] for i in range(nentries) ]
    predictions = [ v2[i] for i in range(nentries) ]
    weights     = [ v3[i] for i in range(nentries) ]

    auc = roc_auc_score(labels,
                        predictions,
                        sample_weight = weights,
                        average = None,
                        )

    for fout in fouts:
        print >> fout
        print >> fout, "%s AUC: %f" % (name, auc)
        fout.flush()

