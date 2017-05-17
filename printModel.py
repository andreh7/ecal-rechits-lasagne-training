#!/usr/bin/env python

import sys

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]

assert len(ARGV) == 2

inputFile, outputFile = ARGV


# try to infer output format
if outputFile.lower().endswith(".pdf"):
    outputFormat = "pdf"
elif outputFile.lower().endswith(".png"):
    outputFormat = "png"
elif outputFile.lower().endswith(".gv"):
    outputFormat = "raw"
else:
    print >> sys.stderr,"can't infer output format from output file",outputFile
    sys.exit(1)

#----------------------------------------

import lasagne.layers

# load the network
import cPickle as pickle

model = pickle.load(open(inputFile))['model']

import draw_net
dot = draw_net.get_pydot_graph(lasagne.layers.get_all_layers(model), verbose = True)

dot.write(outputFile, format = outputFormat)

