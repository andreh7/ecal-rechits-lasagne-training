#!/usr/bin/env python

execfile("model09-bdt-inputs.py")

def makeModel():

    return makeModelHelper(
        numHiddenLayers = 10,
        nodesPerHiddenLayer = 100
        )



