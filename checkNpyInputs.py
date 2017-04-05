#!/usr/bin/env python

import sys, os

import numpy as np
np.set_printoptions(linewidth=500)


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]


# number of photon candidates found with center not one
centerNotOne = 0

duplicateCenterValues = 0


missingCenterValues = 0

total = 0

width, height = 35, 35
wmid, hmid = width / 2 + 1, height / 2 + 1

expectedFirstIndex = 1

for findex,fname in enumerate(ARGV):

    print >> sys.stderr,"opening",fname,"(%d/%d)" % (findex + 1, len(ARGV))

    data = np.load(fname)

    # import tqdm
    # progbar = tqdm.tqdm(total = input1.totNumEvents,
    #                     mininterval = 0.1,
    #                     unit = 'samples',
    #                     )


    firstIndex = data['X/firstIndex']
    numRecHits = data['X/numRecHits']

    xx         = data['X/x']
    yy         = data['X/y']
    energy     = data['X/energy']

    for pos in range(len(firstIndex)):

        if firstIndex[pos] != expectedFirstIndex:
            print "firstIndex is wrong at pos",pos,": found %d, expected %d" % (firstIndex[pos], expectedFirstIndex)

        indices = np.arange(firstIndex[pos], firstIndex[pos] + numRecHits[pos]) - 1
        
        centerValue = None
        hasDuplicateCenterValue = False

        for index in indices:
            if xx[index] == wmid and yy[index] == hmid:
                if centerValue != None:
                    hasDuplicateCenterValue = True
                else:
                    centerValue = energy[index]

            # if abs(energy[index] - 1.) < 1e-7:
            #    print xx[index],yy[index]

        if hasDuplicateCenterValue:
            duplicateCenterValues += 1

        if centerValue == None:
            missingCenterValues += 1
            print "missing center value at pos",pos
        elif abs(centerValue - 1.0) > 1e-7:
            centerNotOne += 1

        expectedFirstIndex += numRecHits[pos]

# print "found",centerNotOne,"out of",total,"(%5.1f %%)" % (100. * centerNotOne / float(total)),"photon candidates with center element not one"

print "duplicateCenterValues=",duplicateCenterValues, "missingCenterValues=",missingCenterValues, "centerNotOne=",centerNotOne
