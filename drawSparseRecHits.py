#!/usr/bin/env python

import sys, os
sys.path.append(os.path.expanduser("~/torchio"))
import torchio

import pylab
import numpy as np

#----------------------------------------------------------------------

def drawIndividual(photonIndices):

    for photonIndex in photonIndices:
        assert photonIndex >= 0 and photonIndex < numPhotons, "photon indices must be in the range 0..%d" % (numPhotons - 1)


    for photonIndex in photonIndices:

        data = np.zeros((width, height))

        # torch coordinates
        XX, YY = np.meshgrid(np.arange(width) + 1.5, np.arange(height) + 1.5)

        # note that the torch indices are one based
        # so we subtract one here
        baseIndex = firstIndices[photonIndex] - 1

        for i in range(numRecHits[photonIndex]):

            xc = xcoords[baseIndex + i] - 1
            yc = ycoords[baseIndex + i] - 1

            assert xc >= 0
            assert xc < width
            assert yc >= 0
            assert yc < height

            data[yc, xc] = energies[baseIndex + i]

        # plot the matrix
        pylab.figure()
        pylab.pcolor(XX, YY, data, cmap = pylab.cm.Blues)

        pylab.title("index " + str(photonIndex) + " (label=" + str(labels[photonIndex]) + ")")
        pylab.grid()


#----------------------------------------------------------------------

def drawSummary(maxEntries = None):
    # draw a projection of all photon candidates, separated
    # by label

    if maxEntries == None:
        maxEntries = len(labels)

    for label in (0, 1):

        print "projecting label",label

        data = np.zeros((width, height))

        numFound = 0

        # torch coordinates
        XX, YY = np.meshgrid(np.arange(width) + 1.5, np.arange(height) + 1.5)

        for photonIndex in range(len(labels)):

            if photonIndex >= maxEntries:
                break

            if labels[photonIndex] != label:
                continue

            numFound += 1

            # note that the torch indices are one based
            # so we subtract one here
            baseIndex = firstIndices[photonIndex] - 1

            for i in range(numRecHits[photonIndex]):

                xc = xcoords[baseIndex + i] - 1
                yc = ycoords[baseIndex + i] - 1

                assert xc >= 0
                assert xc < width
                assert yc >= 0
                assert yc < height

                data[yc, xc] += energies[baseIndex + i]

            # end of loop over rechits of this photon

        # end of loop over photon candidates

        # plot the matrix
        pylab.figure()
        pylab.pcolor(XX, YY, data, cmap = pylab.cm.Blues)

        pylab.title("label=" + str(label) + " (%d candidates)" % numFound)
        pylab.grid()

    # end of loop over labels


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

# parse command line arguments
import argparse

parser = argparse.ArgumentParser(prog='drawSparseRecHits',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                 )


parser.add_argument('--max-entries',
                    metavar = "n",
                    type = int,
                    default = None,
                    help='maximum number of photon candidates to consider when drawing the summary',
                    dest = "maxEntries",
                    )


parser.add_argument('inputFile',
                    metavar = "inputFile",
                    type = str,
                    nargs = 1,
                    help='input file',
                    )

parser.add_argument('indices',
                    metavar = "indices",
                    type = int,
                    nargs = '+',
                    help='indices of photon candidates to be drawn or -1 for summary',
                    )

options = parser.parse_args()
#----------------------------------------


print "loading data..."
inputFname = options.inputFile[0]

if inputFname.endswith('.npz'):
    data = np.load(inputFname)

    firstIndices = data['X/firstIndex']
    numRecHits = data['X/numRecHits']
    xcoords = data['X/x']
    ycoords = data['X/y']
    energies = data['X/energy']
    labels = data['y']

elif inputFname.endswith(".t7"):
    data = torchio.read(inputFname)

    firstIndices = data['X']['firstIndex']
    numRecHits = data['X']['numRecHits']
    xcoords = data['X']['x']
    ycoords = data['X']['y']
    energies = data['X']['energy']
    labels = data['y']


# TODO: read this from the file
width = 35
height = 35


print "done loading data"


numPhotons = len(labels)
print "read",numPhotons,"photons"

photonIndices = options.indices

if photonIndices == [ -1 ]:
    # draw average of all photons, separated by label
    drawSummary(maxEntries = options.maxEntries)
else:
    drawIndividual(photonIndices)

    

pylab.show()
