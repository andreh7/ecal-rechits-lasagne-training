#!/usr/bin/env python

# compares TMVA training output files
# (originally introduced to validate
# our speeding up of the TMVA
# training code)

# needs a ROOT environment which we can't
# make compatible with our lasagne environment

import sys

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]

assert len(ARGV) == 2


import ROOT; gcs = []

for treeName in [ 'TrainTree', 'TestTree' ]:

    # read ROOT files
    for fileIndex, fname in enumerate(ARGV):

        fin = ROOT.TFile(fname)
        assert fin.IsOpen()

        tree = fin.Get(treeName)

        tree.SetEstimate(tree.GetEntries())

        tree.Draw("origindex:BDT","","goff")

        v1 = tree.GetV1()
        v2 = tree.GetV2()
        
        nevents = tree.GetSelectedRows()

        origIndex = [ int(v1[i]) for i in range(nevents) ]

        # assume uniqueness
        assert len(origIndex) == len(set(origIndex))

        # get BDT output values
        bdtValues = [ v2[i] for i in range(nevents) ]

        assert len(bdtValues) == len(origIndex)

        if fileIndex == 0:
            # first file: build a map from event index to BDT output value
            firstFileValues = dict(zip(origIndex, bdtValues))
        
        else:
            # check that the indices actually are the same
            # and find for each BDT value in this file the BDT
            # value of the previous (reference) file

            bdtDiffs = []
            assert len(origIndex) == len(firstFileValues)

            for oi, bdt in zip(origIndex, bdtValues):
                refBdt = firstFileValues[oi]

                bdtDiffs.append(bdt - refBdt)
            


    # plot the difference using ROOT, not matplotlib (in this environment)
                
    # fill a tuple first
    ROOT.gROOT.cd()
    plotTuple = ROOT.TNtuple("plotTuple", "plotTuple", "bdtDiff"); gcs.append(plotTuple)
    for bdtDiff in bdtDiffs:
        plotTuple.Fill(bdtDiff)
    
    canvas = ROOT.TCanvas(); gcs.append(canvas)

    plotTuple.Draw("bdtDiff")
    plotTuple.GetHistogram().SetTitle(treeName)

    ROOT.gPad.SetLogy()

    print "maximum absolute difference (%s):" % treeName,max(bdtDiffs, key = abs)

    del firstFileValues
