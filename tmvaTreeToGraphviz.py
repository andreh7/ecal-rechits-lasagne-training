#!/usr/bin/env python

# produces a graphviz file from a TMVA BDT weights file
# and the given tree indices

#----------------------------------------------------------------------

import os, sys
import cStringIO as StringIO

#----------------------------------------------------------------------

def getTrainOptions(xmlroot):
    trainOptions = {}

    optionsNode = next(xmlroot.iter('Options'))
    for optionNode in optionsNode.iter('Option'):

        optionName = optionNode.attrib['name']
        assert not trainOptions.has_key(optionName)

        trainOptions[optionName] = dict(
            modified = optionNode.attrib['modified'] == 'Yes',

            # leave the option values as strings for the moment
            value = optionNode.text,
            )
    
    return trainOptions


#----------------------------------------------------------------------

def fillAdjList(adjList, indexToNode, thisNodeIndex, xmlNode):
    # assigns node numbers, fills the adjacency list and the indexToNode dict
    children = [ node for node in xmlNode if node.tag == 'Node' ]
    
    indexToNode[thisNodeIndex] = xmlNode
    adjList[thisNodeIndex] = []

    for child in children:
        # assume the children were not yet seen and therefore do not
        # have an index assigned yet
        
        childIndex = len(indexToNode)
        assert childIndex not in indexToNode
        
        adjList[thisNodeIndex].append(childIndex)

        # visit child
        fillAdjList(adjList, indexToNode, childIndex, child)

#----------------------------------------------------------------------

def writeEdges(fout, nodeNamePrefix, adjList, indexToNode, nodeIndex):
    
    for childIndex in adjList[nodeIndex]:

        childNode = indexToNode[childIndex]

        pos = childNode.attrib['pos']

        print >> fout, "%sn%d -> %sn%d" % (nodeNamePrefix, nodeIndex, nodeNamePrefix, childIndex),

        print >> fout,"[",
        if pos == 'l':
            print >> fout,'label="<"',
        elif pos == 'r':
            print >> fout,'label=">="',


        print >> fout,"];"
    
        # recurse
        writeEdges(fout, nodeNamePrefix, adjList, indexToNode, childIndex)

#----------------------------------------------------------------------

def writeTree(fout, nodeNamePrefix, xmlNode, varnames):

    adjList = {}
    indexToNode = {}

    fillAdjList(adjList, indexToNode, 0, xmlNode)

    # write vertices
    for nodeIndex in adjList.keys():
        nodeName = "%sn%d"  % (nodeNamePrefix, nodeIndex)

        attrs = {}
        
        xmlNode = indexToNode[nodeIndex]

        splitVarIndex = int(xmlNode.attrib["IVar"])
        splitValue    = float(xmlNode.attrib["Cut"])

        labels = []

        if splitVarIndex != -1:
            # non-leaf node
            labels.append('%s' % varnames[splitVarIndex])
            labels.append("%f" % splitValue)
        else:
            # leaf node
            purity = float(xmlNode.attrib['purity'])
            res = float(xmlNode.attrib['res'])

            # because we have UseYesNoLeaf == True, evaluation of a single tree
            # is by calling leaf node -> GetNodeType()
            # see https://github.com/root-project/root/blob/51a90f7caad83c849b2e396adbe63f090b9d4a39/tmva/tmva/src/DecisionTree.cxx#L1720
            #
            # however, from empirical tests, it looks like the value which
            # is returned is basically the value in 'res' (even though
            # some of the trailing digits are different in the ROOT tree
            # and in the xml file)

            labels.append('purity=%f' % purity)
            labels.append('res=%f' % res)


        if labels:
            attrs['label'] = '"%s"' % "\\n".join(labels)
        else:
            attrs['label'] = '""'

        #----------
        print >> fout, nodeName,"[",
        print >> fout, ",".join([ "%s=%s" % (key, value) for key, value in attrs.items()]),

        print >> fout, "];"

    # write edges
    writeEdges(fout, nodeNamePrefix, adjList, indexToNode, 0)

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]
assert len(ARGV) >= 2

weightsFile = ARGV.pop(0)

treeIndices = [ int(arg) for arg in ARGV ]

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
# TODO: we should check from the XML file

varnames = [ node.attrib['Label'] for node in xmlroot.iter('Variable') ]

#----------
# this influences how the tree is evaluated
# see https://github.com/root-project/root/blob/51a90f7caad83c849b2e396adbe63f090b9d4a39/tmva/tmva/src/DecisionTree.cxx#L1720
trainOptions = getTrainOptions(xmlroot)
assert trainOptions['UseYesNoLeaf']['value'] == 'True'

#----------

# get the number of trees
numTrees = int(next(xmlroot.iter("Weights")).attrib['NTrees'])

# check that the given tree indices are valid
for treeIndex in treeIndices:
    assert treeIndex >= 0
    assert treeIndex < numTrees

treeIndices.sort()

#----------
assert len(treeIndices) > 0

fout = StringIO.StringIO()
print >> fout,"digraph {"

for treeIndex, treeNode in enumerate(xmlroot.iter("BinaryTree")):

    if treeIndex != treeIndices[0]:
        continue

    treeRoot = [ node for node in treeNode if node.tag == 'Node' ]
    assert len(treeRoot) == 1
    treeRoot = treeRoot[0]

    # walk the tree
    writeTree(fout, "t%04d_" % treeIndex, treeRoot, varnames)

    # prepare next iteration
    treeIndices.pop(0)
    if not treeIndices:
        break

# end of loop over trees

print >> fout, "}"

print fout.getvalue()
