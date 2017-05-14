#!/usr/bin/env python

import sys, os
import lasagne.layers
import cPickle as pickle
import numpy as np



#----------------------------------------------------------------------

class NetworkApplier:
    def __init__(self, modelStructureFile, modelParamsFile, verbose = False):

        self.verbose = verbose

        #----------
        # load model data
        #----------

        modelData = pickle.load(open(modelStructureFile))
        self.model = modelData['model']
        self.input_vars = modelData['input_vars']

        paramsRaw = np.load(modelParamsFile)

        # the keys are of the form arr_<integer>
        params = [ paramsRaw[key] for key in sorted(paramsRaw.keys(), key = lambda x: int(x[4:]) ) ]

        # apply parameters to model
        lasagne.layers.set_all_param_values(self.model, params)

        self.layers = lasagne.layers.get_all_layers(self.model)

        #----------
        # iterate over layers 
        # build a theano function for the outputs of each layer
        #----------
        if self.verbose:

            print >> sys.stderr,"found",len(self.layers),"layers"
            for index, layer in enumerate(self.layers):
                print >> sys.stderr,"  layer %2d:" % index,layer


    #----------------------------------------

    def loadInputs(self, inputsFile):
        #----------
        # load input variables
        #----------

        print >> sys.stderr,"loading input variables"
        inputData = np.load(inputsFile)

        self.inputFieldNames = sorted([ key for key in inputData.keys() if key.startswith('input/') ])

        self.inputData = [ inputData[key] for key in self.inputFieldNames ]

        if self.verbose:
            print >> sys.stderr,"input data shapes:"
            for name, data in zip(self.inputFieldNames, self.inputData):
                print >> sys.stderr, "  %-30s: %s" % (name, str(data.shape))

            # find input layers: assume the input values are given
            # in the same order as the input layers appear fg
            inputLayers = [ layer for layer in self.layers if isinstance(layer, lasagne.layers.input.InputLayer) ]

            print >> sys.stderr,"input layer shapes:"
            for layer in inputLayers:
                print >> sys.stderr, "  %-30s: %s" % (layer.name, str(layer.shape))

    #----------------------------------------

    def apply(self, returnIntermediateValues):

        inputData = self.inputData

        numSamples = inputData[0].shape[0]

        #----------
        # iterate over samples and apply them to the network
        #----------

        layerOutputFunctions = []

        layerOutputValues = []

        import theano

        for index, layer in enumerate(self.layers):
            # if isinstance(layer, lasagne.layers.input.InputLayer):
            #     # skip input layer
            #     continue

            shape = layer.output_shape

            # the first dimension is the minibatch dimension (typically
            # set to None), we ignore it

            print >> sys.stderr, "layer %2d has" % index, np.prod(shape[1:]),"nodes"

            if not returnIntermediateValues and index < len(self.layers) - 1:
                continue

            # produce a theano variable to hold 
            thisLayerOutput = lasagne.layers.get_output(layer, deterministic = True)
            thisLayerOutputFunction = theano.function(self.input_vars, thisLayerOutput)

            layerOutputFunctions.append(thisLayerOutputFunction)

            shape = tuple([ numSamples ] + list(shape[1:]))
            layerOutputValues.append(dict(index = index,
                                          outputVal = np.zeros(shape)))

        #----------
        # loop over input samples
        #----------
        evalBatchSize = 10000

        for start in range(0,numSamples,evalBatchSize):

            end = min(start + evalBatchSize,numSamples)

            for item, outputFunc in zip(layerOutputValues, layerOutputFunctions):

                item['outputVal'][start:end] = outputFunc(
                    *[ inp[start:end] for inp in inputData ]
                )


        return layerOutputValues

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

if __name__ == '__main__':
    ARGV = sys.argv[1:]

    sample = "test"

    resultDir, epoch = ARGV

    epoch      = int(epoch)

    modelFile  = os.path.join(resultDir, "model-structure.pkl")
    paramsFile = os.path.join(resultDir, "model-%04d.npz" % epoch)
    inputsFile = os.path.join(resultDir, "input-%s.npz" % sample)

    #----------
    # calculate the network output values
    #----------
    
    networkApplier = NetworkApplier(modelFile, paramsFile, verbose = True)
    networkApplier.loadInputs(inputsFile)

    layerOutputValues = networkApplier.apply(returnIntermediateValues = False)

    numSamples = len(layerOutputValues[-1]['outputVal'])

    #----------
    # load weights and labels
    #----------
    weightsLabelsFile = os.path.join(resultDir, "weights-labels-" + sample + ".npz")

    if os.path.exists(weightsLabelsFile):
        weightsLabels = np.load(weightsLabelsFile)
    else:
        import bz2
        weightsLabels = np.load(bz2.BZ2File(weightsLabelsFile + ".bz2"))

    if sample == 'train':
        weightVarName = "trainWeight"
    else:
        # test sample
        weightVarName = "weight"

    weights = weightsLabels[weightVarName]
    labels  = weightsLabels['label']

    assert len(labels) == numSamples, "len(labels)=%d, numSamples=%d" % (len(labels), numSamples)
    assert len(weights) == numSamples


    #----------
    # write data out in npz format
    #----------
    # start layer indexing at one because we skipped the input layer
    outputData = dict( [ ("layer_%02d_out" % item['index'], item['outputVal']) for item in layerOutputValues ])
    outputData['weights'] = weights
    outputData['labels'] = labels

    outputName = os.path.join(resultDir, "layerValues-%s-%04d.npz" % (sample, epoch))

    print "writing output data"
    np.savez_compressed(outputName, **outputData)
    print >> sys.stderr,"wrote output to",outputName
