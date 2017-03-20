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
# initialize output directory
#----------

if options.outputDir == None:
    options.outputDir = "results/" + time.strftime("%Y-%m-%d-%H%M%S")

if not os.path.exists(options.outputDir):
    os.makedirs(options.outputDir)

#----------
# try to set the process name
#----------
try:
    import procname
    procname.setprocname("train " + 
                         os.path.basename(options.outputDir.rstrip('/')))
except ImportError, ex:
    pass

#----------
# setup logging
#----------
logfile = open(os.path.join(options.outputDir, "train.log"), "w")

fouts = [ sys.stdout, logfile ]

#----------

print "loading data"

doPtEtaReweighting = globals().get("doPtEtaReweighting", False)

cuda = True
with Timer("loading training dataset...") as t:
    trainData, trsize, trainEventIds = datasetLoadFunction(dataDesc['train_files'], dataDesc['trsize'], 
                                                           cuda = cuda, 
                                                           isTraining = True,
                                                           reweightPtEta = doPtEtaReweighting,
                                                           logStreams = fouts,
                                                           returnEventIds = True)
with Timer("loading test dataset...") as t:
    testData,  tesize, testEventIds = datasetLoadFunction(dataDesc['test_files'], dataDesc['tesize'], cuda, 
                                                          isTraining = False,
                                                          reweightPtEta = False,
                                                          logStreams = fouts,
                                                          returnEventIds = True)

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
# write training file paths to result directory
#----------

fout = open(os.path.join(options.outputDir, "samples.txt"), "w")
for fname in dataDesc['train_files']:
    print >> fout, fname
fout.close()

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

import ROOT; gcs = []
ROOT.TMVA.gConfig().GetIONames().fWeightFileDir = options.outputDir

ROOT.TMVA.Tools.Instance()
tmvaFname = os.path.join(options.outputDir, "tmva.root")
tmvaOutputFile = ROOT.TFile(tmvaFname,"RECREATE")

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

factory.AddSpectator("istrain","F")

# official photon id
factory.AddSpectator("origmva","F")

# index of event in our numpy arrays
factory.AddSpectator("origindex","I")

# weight which we gave to TMVA for training
# (TMVA potentially normalizes the per class sum of weights differently,
# i.e. weights are scaled per class)
factory.AddSpectator("trainWeight","F")
factory.AddSpectator("origTrainWeights","F")

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

    #                                             spectators
    values = ROOT.vector('double')(numInputVars + 5)

    for inputData, labels, weights, origWeights, method, origmva, istrain in (
        (trainInput, trainData['labels'], trainData['weights'], origTrainWeights,    factory.AddTrainingEvent, trainData['mvaid'], 1),
        (testInput,  testData['labels'],  testData['weights'],  testData['weights'], factory.AddTestEvent, testData['mvaid'], 0)):

        for row in range(inputData.shape[0]):

            for index, value in enumerate(inputData[row]):
                values[index] = value
            
            values[numInputVars]   = istrain
            values[numInputVars+1] = origmva[row]

            # origindex
            values[numInputVars+2] = row

            # our training weight
            values[numInputVars+3] = weights[row]

            # original event weight potentially before eta/pt reweighting (for performance comparison)
            values[numInputVars+4] = origWeights[row]

            if labels[row] == 1:
                method("Signal", values, weights[row])
            else:
                method("Background", values, weights[row])


nTrain_Signal     = sum(trainData['labels'] == 1)
nTrain_Background = sum(trainData['labels'] != 1)
nTest_Signal      = sum(testData['labels'] == 1)
nTest_Background  = sum(testData['labels'] != 1)


factory.PrepareTrainingAndTestTree(ROOT.TCut(""),
                                   ",".join([
            "nTrain_Signal=%d" % nTrain_Signal,
            "nTrain_Background=%d" % nTrain_Background,
            "nTest_Signal=%d" % nTest_Signal,
            "nTest_Background=%d" % nTest_Background,
            "SplitMode=Block",
            ]))

#----------
# run training with TMVA
#----------

method = factory.BookMethod(ROOT.TMVA.Types.kBDT, "BDT",
                   ":".join([
            "!V",
            "VerbosityLevel=Default",
            "VarTransform=None",
            "!H",
            "!CreateMVAPdfs",
            "!IgnoreNegWeightsInTraining",
            "NTrees=2000",
            "MaxDepth=6",
            "MinNodeSize=5%",
            "nCuts=2000",
            "BoostType=Grad",
            "AdaBoostR2Loss=quadratic",
            "!UseBaggedBoost",
            "Shrinkage=1.000000e-01",
            "AdaBoostBeta=5.000000e-01",
            "!UseRandomisedTrees",
            "UseNvars=4",
            "UsePoissonNvars",
            "BaggedSampleFraction=6.000000e-01",
            "UseYesNoLeaf",
            "NegWeightTreatment=ignorenegweightsintraining",
            "Css=1.000000e+00",
            "Cts_sb=1.000000e+00",
            "Ctb_ss=1.000000e+00",
            "Cbb=1.000000e+00",
            "NodePurityLimit=5.000000e-01",
            "SeparationType=giniindex",
            "!DoBoostMonitor",
            "!UseFisherCuts",
            "MinLinCorrForFisher=8.000000e-01",
            "!UseExclusiveVars",
            "!DoPreselection",
            "SigToBkgFraction=1.000000e+00",
            "PruneMethod=costcomplexity",
            "PruneStrength=5.000000e+00",
            "PruningValFraction=5.000000e-01",
            "nEventsMin=0",
            "!UseBaggedGrad",
            "GradBaggingFraction=6.000000e-01",
            "UseNTrainEvents=0",
            "NNodesMax=0",
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
tmvaOutputFile = ROOT.TFile(tmvaFname, "UPDATE")

for name, treeName, eventIds in (('train', 'TrainTree', trainEventIds),
                                 ('test', 'TestTree',   testEventIds)):
    
    tree = tmvaOutputFile.Get(treeName)
    ROOT.gROOT.cd()

    assert tree != None, "could not find tree " + treeName

    if tree.GetEntries() > 1000000:
        tree.SetEstimate(tree.GetEntries())

    tree.Draw("classID:BDT:weight:origindex","","goff")
    nentries = tree.GetSelectedRows()

    v1 = tree.GetV1(); v2 = tree.GetV2(); v3 = tree.GetV3(); v4 = tree.GetV4()

    labels      = [ v1[i] for i in range(nentries) ]
    predictions = [ v2[i] for i in range(nentries) ]
    weights     = [ v3[i] for i in range(nentries) ]
    origindex   = [ int(v4[i] + 0.5) for i in range(nentries) ]

    auc = roc_auc_score(labels,
                        predictions,
                        sample_weight = weights,
                        average = None,
                        )

    for fout in fouts:
        print >> fout
        print >> fout, "%s AUC: %f" % (name, auc)
        fout.flush()


    #----------
    # add another tree with sample/run/ls/event numbers
    # (this tree can be friended afterwards)
    #----------

    eventTree = ROOT.TTree(treeName + "Events", treeName + "Events")
    gcs.append(eventTree)

    import array
    # upper case are unsigned
    arrSample = array.array( 'I', [ 0 ] )
    arrRun    = array.array( 'I', [ 0 ] )
    arrLS     = array.array( 'I', [ 0 ] )
    arrEvent  = array.array( 'L', [ 0 ] )

    # lowercase characters are unsigned
    eventTree.Branch('sample', arrSample, 'sample/i') 
    eventTree.Branch('run',    arrRun,    'run/i') 
    eventTree.Branch('ls',     arrLS,     'ls/i') 
    eventTree.Branch('event',  arrEvent,  'event/l') 

    for origInd in origindex:
        arrSample[0] = eventIds['sample'][origInd]
        arrRun[0]    = eventIds['run']   [origInd]
        arrLS[0]     = eventIds['ls']    [origInd]
        arrEvent[0]  = eventIds['event'] [origInd]

        eventTree.Fill()

    tmvaOutputFile.cd()
    eventTree.Write()

ROOT.gROOT.cd()
tmvaOutputFile.Close()

#----------
print "output directory is",options.outputDir
