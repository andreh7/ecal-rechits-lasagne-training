#!/usr/bin/env python

import bdtvarsimportanceutils

#----------
# configuration for BDT variable scan (with shallow neural network)
#  run around 2017-07-01
#----------

# maximum number of epochs for each training
maxEpochs = 200

# how many of the last epochs should be considered
# for the figure of merit calculation ?
windowSize = 10



dataSetFname = "dataset14-bdt-inputvars.py"

modelFname   = "model09-bdt-inputs.py"

additionalOptions = [
    "--param doPtEtaReweighting=True",
    "--param sigToBkgFraction=1.0",
    "--opt sgd",
    ]


commandPartsBuilder = bdtvarsimportanceutils.commandPartsBuilderNN

# how to read the result files
resultFileReader = bdtvarsimportanceutils.ResultFileReaderNN(windowSize, maxEpochs)
