#!/usr/bin/env python

#----------
# configuration for BDT variable scan (with parallelized BDT implementation)
#  run around 2017-09-27
#----------

# maximum number of epochs for each training
maxEpochs = 200

# how many of the last epochs should be considered
# for the figure of merit calculation ?
windowSize = 10



dataSetFname = "dataset14-bdt-inputvars.py"

modelFname   = "model09c-bdt-inputs-tmva.py"

additionalOptions = [
    "--param doPtEtaReweighting=True",
    '--param "absVarNames=[]"',
    '--param normalizeBDTvars=False',
    '--custom-bdt-code',
    ]


commandPartsBuilder = commandPartsBuilderBDT
