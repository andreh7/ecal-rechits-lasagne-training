#!/usr/bin/env python

# script trying to reproducing the track isolation variables
# by hand

# datasetDir = '../data/2016-10-17-vtx-dz'
# datasetDir = '/tmp/aholz/deleteme'
datasetDir = '../data//2017-06-23-vertex-info/deleteme2'

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


class TrackUtil:
    # keeps information about the tracks of all photons and has some helper methods

    def __init__(self, data):

        self.firstIndex = data['tracks/firstIndex']
        self.numTracks  = data['tracks/numTracks']

        self.trackpt = data['tracks/pt']

        self.charge = data['tracks/charge']
        self.pdgId  = data['tracks/pdgId']
        self.trackVtxIndex = data['tracks/vtxIndex']

        self.trackVtxX = data['tracks/vtxX']
        self.trackVtxY = data['tracks/vtxY']
        self.trackVtxZ = data['tracks/vtxZ']

        self.trackEta = data['tracks/etaAtVertex']
        self.trackPhi = data['tracks/phiAtVertex']

        # supercluster cartesian coordinates
        self.scX       = data['phoVars/scX']
        self.scY       = data['phoVars/scY']
        self.scZ       = data['phoVars/scZ']


#----------------------------------------------------------------------

class SinglePhotonTrackUtil:
    # utility class for tracks of a single photon

    #----------------------------------------

    def __init__(self, trackUtil, photonIndex):

        self.numTracks = trackUtil.numTracks[photonIndex]

        self.trackUtil = trackUtil
        self.photonIndex = photonIndex

        self.trackInd = slice(trackUtil.firstIndex[photonIndex],
                              trackUtil.firstIndex[photonIndex] + self.numTracks)

        # quantities of the tracks of this photon
        self.trackpt       = trackUtil.trackpt[self.trackInd]
        self.trackVtxIndex = trackUtil.trackVtxIndex[self.trackInd]
        self.charge        = trackUtil.charge[self.trackInd]
        self.pdgId         = trackUtil.pdgId[self.trackInd]

        self.trackVtxX     = trackUtil.trackVtxX[self.trackInd]
        self.trackVtxY     = trackUtil.trackVtxY[self.trackInd]
        self.trackVtxZ     = trackUtil.trackVtxZ[self.trackInd]

        self.trackEta      = trackUtil.trackEta[self.trackInd]
        self.trackPhi      = trackUtil.trackPhi[self.trackInd]

    #----------------------------------------

    def getSelectedTrackIndices(self, vertexIndex):

        # track selection criteria
        indices = self.trackpt >= 0.1                    # minimum trackpt

        indices = indices & (self.trackVtxIndex == vertexIndex)

        # candidates must be charged
        indices = indices & (self.charge != 0)

        # reject electrons and muons
        indices = indices & (np.abs(self.pdgId) != 11) & (np.abs(self.pdgId) != 13)

        return indices

    #----------------------------------------

    def deltaEtaPhiR(self, indices):
        # returns dEta, dPhi, dR for the selected tracks
        #
        # indices is the list of selected indices
        # for which the calculations should be done

        # calculate supercluster eta and phi with respect to
        # vertices of surviving tracks
        # (TODO: speed up by calculating this only once for the selected
        #        track (not photon) vertex, needs storing indices
        #        to track vertices instead of storing the vertex
        #        for each track)

        refVertex = [
            self.trackVtxX[indices],
            self.trackVtxY[indices],
            self.trackVtxZ[indices],
            ]

        scdx = self.trackUtil.scX[self.photonIndex] - refVertex[0]
        scdy = self.trackUtil.scY[self.photonIndex] - refVertex[1]
        scdz = self.trackUtil.scZ[self.photonIndex] - refVertex[2]

        # recalculate the physics eta and phi of the supercluster
        # with respect to the track vertices
        scPhi = np.arctan2(scdy, scdx)
        scEta = np.arctanh(scdz / np.sqrt(scdx**2 + scdy **2 + scdz ** 2))

        thisDphi = deltaPhi(self.trackPhi[indices], scPhi)
        thisDeta = self.trackEta[indices] - scEta

        thisDr = np.sqrt(thisDphi ** 2 + thisDeta ** 2)

        return thisDeta, thisDphi, thisDr

    #----------------------------------------

    def getKnownVertexIndices(self):
        # returns a list of all known vertex indices for the 
        # tracks looked at for the given photon

        return set(self.trackVtxIndex)

#----------------------------------------------------------------------

def checkVertex(data, numPhotons, 
                refVertexIsoVarname,
                vertexIndexVarname):
    # checks whether we can reproduce the values vertices
    # given by the vertex indices in the given variable name
    # 
    # refVertexIsoVarname is the name of the variable
    # which contains the isolation value calculated 
    # in flashgg

    mySelectedVertexIso = np.zeros(numPhotons, dtype = 'float32')

    # note that relpt is the pt of the track divided by the photon Et
    # so we have to multiply by the photonEt first

    numTracks = data['tracks/numTracks']

    photonVtxX = data['phoVars/phoVertexX']
    photonVtxY = data['phoVars/phoVertexY']
    photonVtxZ = data['phoVars/phoVertexZ']
    photonVtxIndex = data[vertexIndexVarname]


    # for debugging
    vtxDz    = np.ones(numTracks.sum(), dtype = 'float32') * -10000
    dR       = np.ones(numTracks.sum(), dtype = 'float32') * -10000
    dEta     = np.ones(numTracks.sum(), dtype = 'float32') * -10000
    dPhi     = np.ones(numTracks.sum(), dtype = 'float32') * -10000
    accepted = np.zeros(numTracks.sum(), dtype = 'int32')

    # from https://github.com/cms-analysis/flashgg/blob/e2fac35487f23fe05b20160d7b51f34bd06b0660/MicroAOD/python/flashggTkVtxMap_cfi.py#L10
    maxVtxDz = 0.2

    trackUtil = TrackUtil(data)

    import tqdm

    progbar = tqdm.tqdm(total = numPhotons, mininterval = 1.0, unit = 'samples')

    for photonIndex in range(numPhotons):

        sptu = SinglePhotonTrackUtil(trackUtil, photonIndex)

        trackInd = sptu.trackInd

        thisVtxDz   = sptu.trackVtxZ - photonVtxZ[photonIndex]

        vtxDz[trackInd] = thisVtxDz

        indices = sptu.getSelectedTrackIndices(
            photonVtxIndex[photonIndex], # vertex selected for photon
            )


        thisDeta, thisDphi, thisDr = sptu.deltaEtaPhiR(indices)


        dR[trackInd][indices] = thisDr

        dPhi[trackInd][indices] = thisDphi
        dEta[trackInd][indices] = thisDeta

        # new set of indices
        indices2 = (thisDr <= 0.3)   # outer cone size

        indices2 = indices2 & (thisDr >= 0.02)  # inner (veto) cone size


        mySelectedVertexIso[photonIndex] = sptu.trackpt[indices][indices2].sum()

        # note that accepted[trackInd][indices][indices2] = 1 does not work
        accepted[trackInd][indices] = 1 * indices2

        progbar.update()
    
    # end of loop over photons

    progbar.close()


    import pylab
    print "plotting"

    diff = mySelectedVertexIso - data[refVertexIsoVarname]

    reldiff = diff[data[refVertexIsoVarname] != 0] / data[refVertexIsoVarname][data[refVertexIsoVarname] != 0] - 1

    # maximum absolute difference in sum pt
    for maxAbsDiff in (0.1, 1):
        numDiffering = len(diff[np.abs(diff) < maxAbsDiff])
        print "fraction of events within %.1f GeV: %.1f%% (%d out of %d, %d outside)" % (maxAbsDiff,
                                                                 numDiffering / float(len(diff)) * 100.,
                                                                 numDiffering, len(diff), len(diff) - numDiffering
                                                                 )


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



    print "  diff=",diff[index],"ours=",mySelectedVertexIso[index], "flashgg=", data[refVertexIsoVarname][index], "run:ls:event=%d:%d:%d" % (data['run'][index], data['ls'][index], data['event'][index]),"file=",fnames[data['fileIndex'][index]],"index=",index,"relPhotonsOffset=",index - data['fileOffsetPhotons'][index]

    trkInd = makeTrackIndices(data, index)

    print "tracks:"

    for ind in range(trkInd.start, trkInd.stop):

        print "track pt=", trackUtil.trackpt[ind],
        print "accepted=", accepted[ind],
        print "vtxdz:",    vtxDz[ind],
        print "vtxZ:",     trackUtil.trackVtxZ[ind],
        print "vtxIndex:", trackUtil.trackVtxIndex[ind],
        print "eta:",      trackUtil.trackEta[ind],
        print "phi:",      trackUtil.trackPhi[ind],
        print "dr:",       dR[ind],
        print "dphi:",     dPhi[ind],
        print "deta:",     dEta[ind],
        print "charge:",   trackUtil.charge[ind],
        print "pdgId:",    trackUtil.pdgId[ind],
        print

    print "photon:","et=",data['phoVars/phoEt'][index],"sceta=",data['phoIdInput/scEta'][index],"vtxZ=",photonVtxZ[index]
    
    pylab.figure(); pylab.hist(diff, bins = 100); pylab.title('recalculation minus flashgg')
    pylab.figure(); pylab.hist(diff[np.abs(diff) < 0.1], bins = 100); pylab.title('recalculation minus flashgg')


#----------------------------------------------------------------------

def checkSelectedVertex(data, numPhotons):
    # checks whether we can reproduce the values of the selected
    # photon vertex
    # values from flashgg

    checkVertex(data, numPhotons, 
                'phoIdInput/pfChgIso03',
                'phoVars/phoVertexIndex',
                )

#----------------------------------------------------------------------

def checkWorstVertex(data, numPhotons):
    # checks whether we can reproduce the values of the selected
    # photon vertex
    # values from flashgg

    checkVertex(data, numPhotons, 
                'phoIdInput/pfChgIso03worst',
                'phoVars/phoWorstIsoVertexIndex',
                )

#----------------------------------------------------------------------


def checkWorstVertexRecalculating(data, numPhotons):
    # checks whether we can reproduce the values of the worst
    # photon vertex by recalculating the worst vertex,
    # i.e. by looping over all seen vertices 
    # and calculating the isolation with respect to them
    # (quite slow to do this in python)

    myWorstVertexIso = np.zeros(numPhotons, dtype = 'float32')

    # note that relpt is the pt of the track divided by the photon Et
    # so we have to multiply by the photonEt first

    numTracks = data['tracks/numTracks']

    photonVtxX = data['phoVars/phoVertexX']
    photonVtxY = data['phoVars/phoVertexY']
    photonVtxZ = data['phoVars/phoVertexZ']
    photonVtxIndex = data['phoVars/phoVertexIndex']


    # for debugging
    vtxDz    = np.ones(numTracks.sum(), dtype = 'float32') * -10000
    # dR       = np.ones(numTracks.sum(), dtype = 'float32') * -10000
    # dEta     = np.ones(numTracks.sum(), dtype = 'float32') * -10000
    # dPhi     = np.ones(numTracks.sum(), dtype = 'float32') * -10000
    # accepted = np.zeros(numTracks.sum(), dtype = 'int32')

    # from https://github.com/cms-analysis/flashgg/blob/e2fac35487f23fe05b20160d7b51f34bd06b0660/MicroAOD/python/flashggTkVtxMap_cfi.py#L10
    maxVtxDz = 0.2

    trackUtil = TrackUtil(data)

    for photonIndex in range(numPhotons):

        sptu = SinglePhotonTrackUtil(trackUtil, photonIndex)

        trackInd = sptu.trackInd

        thisVtxDz   = sptu.trackVtxZ - photonVtxZ[photonIndex]

        vtxDz[trackInd] = thisVtxDz

        worstIso = -1000

        # see also https://github.com/cms-analysis/flashgg/blob/50f5699dd8e57c6aad4272c3603c13bd04336506/MicroAOD/src/PhotonIdUtils.cc#L105

        for vtxIndex in sptu.getKnownVertexIndices():
            # calculate the isolation with respect to this
            # vertex

            indices = sptu.getSelectedTrackIndices(vtxIndex)

            thisDeta, thisDphi, thisDr = sptu.deltaEtaPhiR(indices)


            # dR[trackInd][indices] = thisDr

            # dPhi[trackInd][indices] = thisDphi
            # dEta[trackInd][indices] = thisDeta

            # new set of indices
            indices2 = (thisDr <= 0.3)   # outer cone size

            indices2 = indices2 & (thisDr >= 0.02)  # inner (veto) cone size

            thisIso = sptu.trackpt[indices][indices2].sum()

            worstIso = max(thisIso, worstIso)

        myWorstVertexIso[photonIndex] = worstIso

        # note that accepted[trackInd][indices][indices2] = 1 does not work
        # accepted[trackInd][indices] = 1 * indices2


    import pylab
    print "plotting"

    diff = myWorstVertexIso - data['phoIdInput/pfChgIso03worst']

    reldiff = diff[data['phoIdInput/pfChgIso03'] != 0] / data['phoIdInput/pfChgIso03'][data['phoIdInput/pfChgIso03'] != 0] - 1

    # maximum absolute difference in sum pt
    for maxAbsDiff in (0.1, 1):
        numDiffering = len(diff[np.abs(diff) < maxAbsDiff])
        print "fraction of events within %.1f GeV: %.1f%% (%d out of %d, %d outside)" % (maxAbsDiff,
                                                                 numDiffering / float(len(diff)) * 100.,
                                                                 numDiffering, len(diff), len(diff) - numDiffering
                                                                 )


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



    print "  diff=",diff[index],"ours=",myWorstVertexIso[index], "flashgg=", data['phoIdInput/pfChgIso03'][index], "run:ls:event=%d:%d:%d" % (data['run'][index], data['ls'][index], data['event'][index]),"file=",fnames[data['fileIndex'][index]],"index=",index,"relPhotonsOffset=",index - data['fileOffsetPhotons'][index]

    trkInd = makeTrackIndices(data, index)

    print "tracks:"

    for ind in range(trkInd.start, trkInd.stop):

        print "track pt=", trackUtil.trackpt[ind],
        print "accepted=", accepted[ind],
        print "vtxdz:",    vtxDz[ind],
        print "vtxZ:",     trackUtil.trackVtxZ[ind],
        print "vtxIndex:", trackUtil.trackVtxIndex[ind],
        print "eta:",      trackUtil.trackEta[ind],
        print "phi:",      trackUtil.trackPhi[ind],
        print "dr:",       dR[ind],
        print "dphi:",     dPhi[ind],
        print "deta:",     dEta[ind],
        print "charge:",   trackUtil.charge[ind],
        print "pdgId:",    trackUtil.pdgId[ind],
        print

    print "photon:","et=",data['phoVars/phoEt'][index],"sceta=",data['phoIdInput/scEta'][index],"vtxZ=",photonVtxZ[index]
    
    pylab.figure(); pylab.hist(diff, bins = 100); pylab.title('recalculation minus flashgg')
    pylab.figure(); pylab.hist(diff[np.abs(diff) < 0.1], bins = 100); pylab.title('recalculation minus flashgg')

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

import pylab


#----------

numPhotons = len(data['tracks/firstIndex'])

#----------
# calculate sum of pts of tracks per event from the vertex with matching vertex index
#----------
if True:
    checkSelectedVertex(data, numPhotons)

pylab.show()
