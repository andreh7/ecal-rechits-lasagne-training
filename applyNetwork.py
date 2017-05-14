#!/usr/bin/env python

import sys, os
import lasagne.layers
import cPickle as pickle
import numpy as np



#----------------------------------------------------------------------

def applyNetwork(modelStructureFile, modelParamsFile, inputsFile):

    #----------
    # load model data
    #----------

    modelData = pickle.load(open(modelStructureFile))
    model = modelData['model']
    input_vars = modelData['input_vars']

    paramsRaw = np.load(modelParamsFile)

    # the keys are of the form arr_<integer>
    params = [ paramsRaw[key] for key in sorted(paramsRaw.keys(), key = lambda x: int(x[4:]) ) ]

    # apply parameters to model
    lasagne.layers.set_all_param_values(model, params)

    layers = lasagne.layers.get_all_layers(model)

    #----------
    # load input variables
    #----------

    print >> sys.stderr,"loading input variables"
    inputData = np.load(inputsFile)

    inputFieldNames = sorted([ key for key in inputData.keys() if key.startswith('input/') ])

    inputData = [ inputData[key] for key in inputFieldNames ]

    numSamples = inputData[0].shape[0]

    #----------
    # iterate over layers 
    # build a theano function for the outputs of each layer
    #----------
    print >> sys.stderr,"found",len(layers),"layers"

    #----------
    # iterate over samples and apply them to the network
    #----------

    layerOutputFunctions = []

    layerOutputValues = []

    import theano

    for index, layer in enumerate(layers):
        # if isinstance(layer, lasagne.layers.input.InputLayer):
        #     # skip input layer
        #     continue

        shape = layer.output_shape

        # the first dimension is the minibatch dimension (typically
        # set to None), we ignore it

        print >> sys.stderr, "layer %2d has" % index, np.prod(shape[1:]),"nodes"

        # produce a theano variable to hold 

        thisLayerOutput = lasagne.layers.get_output(layer, deterministic = True)
        thisLayerOutputFunction = theano.function(input_vars, thisLayerOutput)

        layerOutputFunctions.append(thisLayerOutputFunction)

        shape = tuple([ numSamples ] + list(shape[1:]))
        layerOutputValues.append(np.zeros(shape))


    #----------
    # loop over input samples
    #----------
    evalBatchSize = 10000

    for start in range(0,numSamples,evalBatchSize):

        end = min(start + evalBatchSize,numSamples)

        for outputVal, outputFunc in zip(layerOutputValues, layerOutputFunctions):

            outputVal[start:end] = outputFunc(
                inputData[start:end]
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
    
    layerOutputValues = applyNetwork(modelFile, paramsFile, inputsFile)

    numSamples = len(layerOutputValues[-1])

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
    outputData = dict( [ ("layer_%02d_out" % index, outputVal) for index, outputVal in enumerate(layerOutputValues) ])
    outputData['weights'] = weights
    outputData['labels'] = labels

    outputName = os.path.join(resultDir, "layerValues-%s-%04d.npz" % (sample, epoch))

    print "writing output data"
    np.savez_compressed(outputName, **outputData)
    print >> sys.stderr,"wrote output to",outputName




