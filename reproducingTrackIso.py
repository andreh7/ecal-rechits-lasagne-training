#!/usr/bin/env python

# script trying to reproducing the track isolation variables
# by hand

# datasetDir = '../data/2016-10-17-vtx-dz'
datasetDir = '/tmp/aholz/deleteme'

#----------------------------------------------------------------------

import numpy as np
import glob

#----------------------------------------------------------------------

def makeTrackIndices(data, photonIndex):
    return slice(data['tracks/firstIndex'][photonIndex], data['tracks/firstIndex'][photonIndex] + data['tracks/numTracks'][photonIndex])

def deltaPhi(phi1, phi2):
    dphi = phi1 - phi2

    import math

    # will also bring negative values to 0..2pi
    dphi = np.mod(dphi, 2 * math.pi)

    # if dphi > math.pi:
    #    dphi = 2 * math.pi - dphi
    dphi = np.min([dphi, 2 * math.pi - dphi], axis = 0)

    return dphi

#----------------------------------------------------------------------

def deltaR(obj1, obj2):

    dphi = deltaPhi(obj1.phi(), obj2.phi())

    deta = math.fabs(obj1.eta() - obj2.eta())

    dr = math.sqrt(deta * deta + dphi * dphi)

    return dr

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

# load and combine files: note that the 'firstIndices'
# array must be modified

data = {}

tracksOffset = 0
photonsOffset = 0

# for fname in glob.glob(datasetDir + "/*-barrel-train.npz"):

fnames = glob.glob(datasetDir + "/*-barrel*.npz")

data['fileIndex'] = []
data['fileOffsetPhotons'] = []

for fileIndex, fname in enumerate(fnames):

    print "opening file %d/%d" % (fileIndex + 1, len(fnames))
    
    thisData = np.load(fname)
    thisNumTracks = None

    thisNumPhotons = len(thisData['event'])

    data['fileIndex'].append(np.ones(thisNumPhotons, dtype='int32') * fileIndex)
    data['fileOffsetPhotons'].append(np.ones(thisNumPhotons, dtype='int32') * photonsOffset)

    for key, values in thisData.items():
        
        if key == 'tracks/firstIndex':
            # go from torch 1-based indexing to python 0-based indexing
            assert values[0] == 1
            values = values - 1

            values += tracksOffset

        data.setdefault(key, []).append(values)

    # end of loop over keys in the current file

    # prepare next iteration

    thisNumTracks = thisData['tracks/numTracks'].sum()

    tracksOffset += thisNumTracks
    photonsOffset += thisNumPhotons

    # DEBUG
    # break
    
# end of loop over files
        
# now concatenate data from individual files
for key in data.keys():

    data[key] = np.concatenate(data[key])

#----------
# check track indices
#----------
if True:
    prevSum = 0

    firstIndex = data['tracks/firstIndex']
    numTracks = data['tracks/numTracks']

    for index in range(len(firstIndex)):
        assert prevSum == firstIndex[index],"index=" + str(index) + " prevSum=" + str(prevSum) + " firstIndex=" + str(firstIndex[index])
        prevSum += numTracks[index]
        

    
#----------

if False:
    # photon indices
    index = np.where((data['run'] == 1) & (data['ls'] == 18977) & (data['event'] == 55774237))[0][0]

    print "index=",index

    trackpt = data['tracks/pt']

    trkInd = makeTrackIndices(data, index)
    print "trkInd=",trkInd

    print "ZZZ",trackpt[trkInd]


# typical keys:
#   ['X/firstIndex',
#    'X/numRecHits',
#    'X/energy',
#    'X/x',
#    'X/y',
#    'run',
#    'ls',
#    'event',
#    'y',
#    'weight',
#    'mvaid',
#    'genDR',
#    'chgIsoWrtChosenVtx',
#    'chgIsoWrtWorstVtx',
#    'phoIdInput/scRawE',
#    'phoIdInput/r9',
#    'phoIdInput/covIEtaIEta',
#    'phoIdInput/phiWidth',
#    'phoIdInput/etaWidth',
#    'phoIdInput/covIEtaIPhi',
#    'phoIdInput/s4',
#    'phoIdInput/pfPhoIso03',
#    'phoIdInput/pfChgIso03',
#    'phoIdInput/pfChgIso03worst',
#    'phoIdInput/scEta',
#    'phoIdInput/rho',
#    'phoIdInput/esEffSigmaRR',
#    'phoVars/phoEt',
#    'phoVars/phoPhi',
#    'phoVars/diphoMass',
#    'tracks/firstIndex',
#    'tracks/numTracks',
#    'tracks/pt',
#    'tracks/detaAtVertex',
#    'tracks/dphiAtVertex',
#    'tracks/charge',
#    'tracks/vtxDz']


#----------

numEvents = len(data['tracks/firstIndex'])

#----------
# calculate sum of pts of tracks per event from the vertex with dz = 0
#----------
if True:
    # values from flashgg
    mySelectedVertexIso = np.zeros(numEvents, dtype = 'float32')

    # note that relpt is the pt of the track divided by the photon Et
    # so we have to multiply by the photonEt first

    firstIndex = data['tracks/firstIndex']
    numTracks = data['tracks/numTracks']

    trackpt = data['tracks/pt']
    trackVtxX = data['tracks/vtxX']
    trackVtxY = data['tracks/vtxY']
    trackVtxZ = data['tracks/vtxZ']

    trackEta = data['tracks/etaAtVertex']
    trackPhi = data['tracks/phiAtVertex']

    # supercluster cartesian coordinates
    scX      = data['phoVars/scX']
    scY      = data['phoVars/scY']
    scZ      = data['phoVars/scZ']

    photonVtxZ = data['phoVars/phoVertexZ']

    charge = data['tracks/charge']
    pdgId  = data['tracks/pdgId']

    # for debugging
    accepted = np.zeros(numTracks.sum(), dtype = 'int32')

    # from https://github.com/cms-analysis/flashgg/blob/e2fac35487f23fe05b20160d7b51f34bd06b0660/MicroAOD/python/flashggTkVtxMap_cfi.py#L10
    maxVtxDz = 0.2

    for photonIndex in range(numEvents):

        trackInd = makeTrackIndices(data, photonIndex)

        thisTrackpt = trackpt[trackInd]
        thisVtxDz   = trackVtxZ[trackInd] - photonVtxZ[photonIndex]

        # track selection criteria
        indices = thisTrackpt >= 0.1                    # minimum trackpt
        indices = indices & (np.abs(thisVtxDz) < maxVtxDz)  # from selected vertex

        # candidates must be charged
        indices = indices & (charge[trackInd] != 0)

        # reject electrons and muons
        indices = indices & (np.abs(pdgId[trackInd]) != 11) & (np.abs(pdgId[trackInd]) != 13)

        # calculate supercluster eta and phi with respect to
        # vertices of surviving tracks
        # (TODO: speed up by calculating this only once for the selected
        #        track (not photon) vertex, needs storing indices
        #        to track vertices instead of storing the vertex
        #        for each track)


        refVertex = [
            trackVtxX[trackInd][indices],
            trackVtxY[trackInd][indices],
            trackVtxZ[trackInd][indices],
            ]

        scdx = scX[photonIndex] - refVertex[0]
        scdy = scY[photonIndex] - refVertex[1]
        scdz = scZ[photonIndex] - refVertex[2]

        scPhi = np.arctan2(scdy, scdx)
        scEta = np.arctanh(scdz / np.sqrt(scdx**2 + scdy **2 + scdz ** 2))

        thisDr = np.sqrt(deltaPhi(scPhi, trackPhi[trackInd][indices]) ** 2 +
                         (scEta - trackEta[trackInd][indices]) ** 2)

        # new set of indices
        indices2 = (thisDr <= 0.3)   # outer cone size

        indices2 = indices2 & (thisDr >= 0.02)  # inner (veto) cone size


        mySelectedVertexIso[photonIndex] = thisTrackpt[indices][indices2].sum()

        # note that accepted[trackInd][indices][indices2] = 1 does not work
        accepted[trackInd][indices] = 1 * indices2


    import pylab
    print "plotting"

    diff = mySelectedVertexIso - data['phoIdInput/pfChgIso03']

    reldiff = diff[data['phoIdInput/pfChgIso03'] != 0] / data['phoIdInput/pfChgIso03'][data['phoIdInput/pfChgIso03'] != 0] - 1

    # maximum absolute difference in sum pt
    for maxAbsDiff in (0.1, 1):
        print "fraction of events within %.1f GeV: %.1f%%" % (maxAbsDiff,
                                                              len(diff[np.abs(diff) < maxAbsDiff]) / float(len(diff)) * 100.)


    # print events with worst agreement
    print "worst agreement:", diff[np.argmax(np.abs(diff))]
    print "worst agreement photons"
    agreementIndices = np.argsort(np.abs(diff))
    index = agreementIndices[-1]

    # DEBUG
    # index = (data['run'] == 1) & ( data['ls'] == 18977) & (data['event'] == 55774237)

    # DEBUG
    # largest flashgg charged isolation
    # index = 575249
    # index = 10559



    print "  diff=",diff[index],"ours=",mySelectedVertexIso[index], "flashgg=", data['phoIdInput/pfChgIso03'][index], "run/ls/event=%d/%d/%d" % (data['run'][index], data['ls'][index], data['event'][index]),"file=",fnames[data['fileIndex'][index]],"index=",index,"relPhotonsOffset=",index - data['fileOffsetPhotons'][index]

    trkInd = makeTrackIndices(data, index)

    print "track pts:",trackpt[trkInd]
    print "track vtxdz:",vtxDz[trkInd]
    print "track dr:",dR[trkInd]
    print "track charge:",charge[trkInd]

    print "photon:","et=",data['phoVars/phoEt'][index],"sceta=",data['phoIdInput/scEta'][index]
    
    pylab.figure(); pylab.hist(diff, bins = 100); pylab.title('recalculation minus flashgg')
    pylab.figure(); pylab.hist(diff[np.abs(diff) < 0.1], bins = 100); pylab.title('recalculation minus flashgg')


pylab.show()
