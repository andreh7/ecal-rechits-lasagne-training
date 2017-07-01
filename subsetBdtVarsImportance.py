#!/usr/bin/env python

# attempts to infer the variable importance on a subset 
# of the sample or scan vs. the false positive (background
# efficiency) rate

trainDir = "finished-results/2017-03-20-145015"

# is the variable importance in the log file determined 
# from the test or training set ?
treeName = "TrainTree"

outputVarname = "BDT"

# maximum number of parallel processes
numProcesses = 24

# maximum background efficiency / false positive rate to allow

#----------------------------------------------------------------------

import os

weightsFile = os.path.join(trainDir, "TMVAClassification_BDT.weights.xml")
rootTupleFile = os.path.join(trainDir, "tmva.root")

#----------------------------------------------------------------------

import numpy as np
import multiprocessing


def readTree(tree, varnames, dtype = 'float32'):

    numEntries = tree.GetEntries()

    if numEntries > tree.GetEstimate():
        tree.SetEstimate(numEntries)

    numVars = len(varnames)

    result = np.ndarray((numEntries, numVars), dtype = dtype)

    for varIndex, varname in enumerate(varnames):

        thisVarValues = result[:,varIndex]

        tree.Draw(varname,"","goff")
        v1 = tree.GetV1()

        for row in range(numEntries):
            thisVarValues[row] = v1[row]

    return result


#----------------------------------------------------------------------

def singleTreeVariableImportances(xmlNode, varImportances, weights, labels, inputVars, separationFunction, nodeSeparation = None):
    # xmlNode is the xmlnode pointing to a Node element from the xml weights file

    # calculate the separation index for this node

    children = [ node for node in xmlNode if node.tag == 'Node' ]

    if not children:
        return

    assert len(children) == 2

    # we have children, calculate the separation gain
    if nodeSeparation == None:
        nodeSeparation = separationFunction(weights[labels == 1].sum(),
                                            weights[labels == 0].sum())

    splitVarIndex = int(xmlNode.attrib["IVar"])
    splitValue    = float(xmlNode.attrib["Cut"])

    splitVar = inputVars[:,splitVarIndex]
        
    # calculate the separation index of this node first
    childIndices  = (splitVar < splitValue, 
                     splitVar >= splitValue)

    childSeparations = []

    for childIndices, childNode in zip(childIndices, children):
        
        childLabels = labels[childIndices]
        childWeights = weights[childIndices]

        childSeparation = separationFunction(childWeights[childLabels == 1].sum(),
                                             childWeights[childLabels == 0].sum())

        childSeparations.append(childSeparation)

        # recurse
        singleTreeVariableImportances(childNode, varImportances, childWeights, childLabels,
                                      inputVars[childIndices], separationFunction, childSeparation)

    # loop over children

    # calculate separation gain of this node
    # see https://github.com/root-project/root/blob/4cac5a12f98eebc39e9b9888ab6b11b40cddf09d/tmva/tmva/src/SeparationBase.cxx#L93
    separationGain = nodeSeparation - sum(childSeparations)

    # update variable importance
    # see https://github.com/root-project/root/blob/d51a0776c68b67baa37ed163a7c41f6a2d3a53b3/tmva/tmva/src/DecisionTree.cxx#L1669
    varImportances[splitVarIndex] += (separationGain * weights.sum())**2


#----------------------------------------------------------------------
def giniSeparation(sumSig, sumBkg):
    # Gini index
    # 
    # see https://github.com/root-project/root/blob/8f87cc3fdf0d697a51266ca90c80885596c3e5d3/tmva/tmva/src/GiniIndex.cxx#L72

    return 2*sumSig * sumBkg/(float(sumSig + sumBkg)**2)

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

# TODO: we should check from the XML file

#----------------------------------------

#----------
# read the BDT weights file
#----------
import xml.etree.ElementTree as ET

xmldoc = ET.parse(weightsFile)
xmlroot = xmldoc.getroot()

#----------
# get the variable names from the BDT file
#----------
# assume the variables are ordered by their index
varnames = [ node.attrib['Label'] for node in xmlroot.iter('Variable') ]

numVars = len(varnames)

#----------
# read the ROOT tree (input variables, classifier output and labels)
#----------

import ROOT
fin = ROOT.TFile(rootTupleFile)
assert fin.IsOpen()

tree = fin.Get(treeName)
assert tree is not None


# read the input variables
inputValues = readTree(tree, varnames)

# read the labels (which are 0 or 1)
labels = readTree(tree, ["classID"], dtype = 'int32')[:,0]

# read the weights
weights = readTree(tree, ["weight"])[:,0]

# read the classifiers output
bdtOutput = readTree(tree, [outputVarname])[:,0]


# restrict the false positive rate (later)

# get the number of trees
numTrees = int(next(xmlroot.iter("Weights")).attrib['NTrees'])


# recalculate the variable importances for each tree

# for gradient boosting, all tree weights are the same (one)
# treeWeights = np.array(numTrees, dtype = 'float32')

import tqdm
progbar = tqdm.tqdm(total = numTrees, unit = 'trees')

#----------

treeRoots = []

for treeNode in xmlroot.iter("BinaryTree"):
    treeRoot = [ node for node in treeNode if node.tag == 'Node' ]
    assert len(treeRoot) == 1
    treeRoot = treeRoot[0]
    
    treeRoots.append(treeRoot)

completionQueue = multiprocessing.Queue()

#----------

def helper(args):
    # use individual vectors to reduce roundoff errors during
    # summing and to allow to run this in parallel
    
    treeRoot = args

    thisVarImportances = np.zeros(numVars, dtype = 'float64')
    
    singleTreeVariableImportances(treeRoot, thisVarImportances, weights, labels, inputValues, giniSeparation)        

    completionQueue.put(1)

    return thisVarImportances

#----------

workerPool = multiprocessing.Pool(processes = numProcesses) 


result = workerPool.map_async(helper,
                              treeRoots,
                              )


# update progress bar whenever a task finishes
numRemaining = len(treeRoots)
while numRemaining > 0:
    completionQueue.get()
    progbar.update()

    numRemaining -= 1

# end of loop over trees
workerPool.close()
workerPool.join()

varImportances = result.get()



progbar.close()


varImportances = sum(varImportances)

# see https://github.com/root-project/root/blob/d51a0776c68b67baa37ed163a7c41f6a2d3a53b3/tmva/tmva/src/MethodBDT.cxx#L2526
varImportances = np.sqrt(varImportances)
varImportances /= varImportances.sum()

# print "varImportances=",varImportances

# print by decreasing order of importance
for index in np.argsort(varImportances)[::-1]:
    print "%-20s: %.3e" % (varnames[index], varImportances[index])
