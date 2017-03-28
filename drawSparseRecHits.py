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
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]

assert len(ARGV) >= 2

inputFname = ARGV.pop(0)

print "loading data..."
data = np.load(inputFname)

# TODO: read this from the file
width = 35
height = 35

firstIndices = data['X/firstIndex']
numRecHits = data['X/numRecHits']
xcoords = data['X/x']
ycoords = data['X/y']
energies = data['X/energy']
labels = data['y']

print "done loading data"


numPhotons = len(labels)
print "read",numPhotons,"photons"

photonIndices = [ int(photonIndex) for photonIndex in ARGV ]

if photonIndices == [ -1 ]:
    # draw average of all photons, separated by label
    drawSummary()
else:
    drawIndividual(photonIndices)

    

pylab.show()
