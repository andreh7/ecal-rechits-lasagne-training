#!/usr/bin/env python

import math

#----------------------------------------------------------------------

# parameters for track histograms
trkAbsDetaMax = 0.4
trkAbsDphiMax = 0.4

trkDetaBinWidth, trkDphiBinWidth = 2 * math.pi / 360.0, 2 * math.pi / 360.0

# for radial histogram
trkDrMax = 0.4
trkDrBinWidth = 2 * math.pi /360.0

#----------------------------------------------------------------------

# like numpy.arange but including the upper end
def myArange(start, stop, step):
    value = start
    while True:
        yield value
        
        if value >= stop:
            break

        value += step

#----------------------------------------------------------------------

def makeSymmetricBinning(maxVal, step):
    bins = list(myArange(0, maxVal, step))

    # symmetrize
    # but avoid doubling the zero value
    return [ -x for x in bins[::-1][:-1]] + bins

#----------------------------------------------------------------------

# note that we do NOT need makeSymmetricBinning(..) here
# because dr does not go negative
trkBinningDr = list(myArange(0, trkDrMax, trkDrBinWidth))
    
trkBinningDeta = makeSymmetricBinning(trkAbsDetaMax, trkDetaBinWidth)
trkBinningDphi = makeSymmetricBinning(trkAbsDphiMax, trkDphiBinWidth)
