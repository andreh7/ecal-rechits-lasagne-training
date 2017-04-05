#!/usr/bin/env python

# reformats rechits
# 
# we change the format for the rechits here: instead of writing 
# out a table of tables, we write out:
#     
#   - a flat tensor with ALL x, y and energy values (three tensors)
#   - an additional tensor mapping from the photon index
#     to the first index in the above tensors
#   - an additional tensor mapping from the photon index
#     to the number of rechits in this photon (as a convenience)

import numpy as np
import os, re

#----------------------------------------------------------------------

def addSparseRecHits(allData, thisData):
    # no need to convert tensor types with npy (these are already correct)
    #
    # TODO: it may be faster to concatenate all files in once

    xkeys = [ key for key in allData.keys() if key.startswith("X/") ]

    if not xkeys:
  
        # this is the first time we add rechits to allData, just
        # copy the vectors over
        #
        # note that we need to copy the values as 
        # we probably can't add to opened .npz file data
        for key in thisData.keys():
            if key.startswith("X/"):
                allData[key] = thisData[key]
    else:
        assert allData['X/numRecHits'].sum() == len(allData['X/energy'])
        assert len(allData['X/firstIndex']) == len(allData['X/numRecHits'])
        assert allData['X/firstIndex'][-1] + allData['X/numRecHits'][-1] - 1 == len(allData['X/energy'])
  
        # append the values for x, y, energy and numRecHits 
        # we must add an offset to firstIndex
        numPhotonsBefore     = len(allData['X/firstIndex'])
        numRecHitsBefore     = len(allData['X/energy'])
  
        thisNumPhotons       = len(thisData['X/firstIndex'])
        thisNumRecHits       = len(thisData['X/energy'])
  
        assert thisData['X/firstIndex'][-1] + thisData['X/numRecHits'][-1] - 1 == thisNumRecHits
  
        allData['X/x']          = np.concatenate([allData['X/x'],          thisData['X/x']])
        allData['X/y']          = np.concatenate([allData['X/y'],          thisData['X/y']])
        allData['X/energy']     = np.concatenate([allData['X/energy'],     thisData['X/energy']])
        allData['X/numRecHits'] = np.concatenate([allData['X/numRecHits'], thisData['X/numRecHits']])
  
        # expand the firstIndex field
        allData['X/firstIndex'] = np.concatenate([allData['X/firstIndex'], np.zeros(thisNumPhotons, dtype='int32')])
  
        # for sanity checks
        expectedFirstIndex = 1
  
        assert thisData['X/firstIndex'][-1] + thisData['X/numRecHits'][-1] - 1 == len(thisData['X/energy'])
  
        for i in range (thisNumPhotons):
            # sanity check of input data
            assert thisData['X/firstIndex'][i] == expectedFirstIndex
    
            assert thisData['X/numRecHits'][i] >= 1
    
            if i < thisNumPhotons - 1:
                assert thisData['X/firstIndex'][i] + thisData['X/numRecHits'][i] == thisData['X/firstIndex'][i+1]
            else:
                assert thisData['X/firstIndex'][i] + thisData['X/numRecHits'][i] - 1 == len(thisData['X/energy']), \
                    str(thisData['X/firstIndex'][i] + thisData['X/numRecHits'][i] - 1) + " " + str(len(thisData['X/energy']))
    
            # add original firstIndex field
            # TODO: we could use a np vector operation here
            allData['X/firstIndex'][numPhotonsBefore + i] = thisData['X/firstIndex'][i] + numRecHitsBefore
    
            expectedFirstIndex = expectedFirstIndex + thisData['X/numRecHits'][i]
  
        # end of loop over photons
  
    # end -- if first time

#----------------------------------------------------------------------

def catItem(item1, item2):
    # note that in the numpy version we do not have nested
    # dicts (in the torch version we had nested tables)

    return np.concatenate([item1, item2])

#----------------------------------------------------------------------

def addTracks(allData, thisData):
    # convert some tensors (to avoid roundoff errors with indices)
  
    tracksKeys = [ key for key in allData.keys() if key.startswith("tracks/") ]

    if not tracksKeys:
  
        # this is the first time we add rechits to allData, just
        # copy the vectors over
  
        for key in thisData.keys():
            if key.startswith("tracks/"):                                                                         
                allData[key] = thisData[key]
    else:
        assert allData['tracks/numTracks'].sum() == len(allData['tracks/relpt'])
        assert allData['tracks/firstIndex'][-1] + allData['tracks/numTracks'][-1] - 1 == len(allData['tracks/relpt'])
  
        # append the values for relpt, charge etc.
        # we must add an offset to firstIndex
        numPhotonsBefore     = len(allData['tracks/firstIndex'])
        numTracksBefore      = len(allData['tracks/relpt'])
  
        thisNumPhotons       = len(thisData['tracks/firstIndex'])
        thisNumRecHits       = len(thisData['tracks/relpt'])
  
        assert thisData['tracks/firstIndex'][thisNumPhotons - 1] + thisData['tracks/numTracks'][thisNumPhotons - 1] - 1 == thisNumRecHits
  
        # concatenate data fields
  
        for varname in thisData.keys():
            if not varname.startswith('tracks/'):
                continue

            if varname != 'tracks/numTracks' and varname != 'tracks/firstIndex':
                allData[varname] = np.concatenate([ allData[varname], thisData[varname] ])

        # end of loop over variables
  
        allData['tracks/numTracks'] = np.concatenate([allData['tracks/numTracks'], thisData['tracks/numTracks']])
  
        #----------
        # expand the firstIndex field
        #----------
        allData['tracks/firstIndex'] = np.concatenate([allData['tracks/firstIndex'], np.zeros(thisNumPhotons, dtype='int32')])
  
        # for sanity checks
        expectedFirstIndex = 1
  
        assert thisData['tracks/firstIndex'][thisNumPhotons - 1] + thisData['tracks/numTracks'][thisNumPhotons - 1] - 1 == len(thisData['tracks/relpt'])
  
        for i in range(thisNumPhotons):
            # sanity check of input data
            assert thisData['tracks/firstIndex'][i] == expectedFirstIndex
    
            # note that we may have photons without any track nearby
            # (this is NOT the case for rechits on the other hand)
            assert thisData['tracks/numTracks'][i] >= 0
    
            if i < thisNumPhotons - 1:
                assert thisData['tracks/firstIndex'][i] + thisData['tracks/numTracks'][i] == thisData['tracks/firstIndex'][i+1]
            else:
                assert thisData['tracks/firstIndex'][i] + thisData['tracks/numTracks'][i] - 1 == len(thisData['tracks/relpt']),  \
                 str(thisData['tracks/firstIndex'][i] + thisData['tracks/numTracks'][i] - 1)  + " " + str(thisData['tracks/relpt'])
    
            # add original firstIndex field
            allData['tracks/firstIndex'][numPhotonsBefore + i] = thisData['tracks/firstIndex'][i] + numTracksBefore
    
            expectedFirstIndex = expectedFirstIndex + thisData['tracks/numTracks'][i]
    
        # end -- loop over photons
  
    # end -- if first time


#----------------------------------------------------------------------

def makeSampleId(fname, numEntries):
    # @return a number determined from the file name 
    fname = os.path.basename(fname)

    fname = fname.split('_')[0]

    num = "".join([ ch for ch in fname if ch >= '0' and ch <= '9' ])

    if num == "":
        num = -1
    else:
        num = int(num, 10)

    return np.ones(numEntries, dtype = 'i4') * num


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
from optparse import OptionParser
parser = OptionParser("""

  usage: %prog [options] data-directory

          merges npz output files from TorchDumper from multiple jobs into single files

    """
    )

(options, ARGV) = parser.parse_args()

if len(ARGV) != 1:
    print >> sys.stderr, "must specify exactly one data directory to work on"
    sys.exit(1)

dataDir = ARGV.pop(0)

#----------

for subdet in ("barrel", "endcap"):

    #----------
    # find all .npz files matching a given pattern and group them
    #----------
  
    # maps from basename to list of files
    fileGroups = {}
  
    # example name: output4/GJet40toInf_rechits-endcap_96.t7
    import glob
    inputFnames = glob.glob(dataDir + "/*-" + subdet + '_*.npz')
  
    if not inputFnames:
        print "no input files found for " + subdet + " in " + dataDir
        continue
  
    for fname in inputFnames:
  
        mo = re.search("/(.*)_rechits-" + subdet + "_(\d+)\.npz$", fname)
        assert mo, "unexpected filename format " + fname

        baseName = mo.group(1)
        number = int(mo.group(2))
    
        fileGroups.setdefault(baseName, {})
    
        assert(not fileGroups[baseName].has_key(number))
        fileGroups[baseName][number] = fname
  
    # end of loop over input file names
  
    #----------
  
    for baseName, fileNames in fileGroups.items():
  
        outputFname = os.path.join(dataDir, baseName + "_rechits-" + subdet + ".npz")
        print "merging files into",outputFname
    
        #----------
        # traverse the list increasing file index
        #----------
        allData = None
    
        for fileIndex in sorted(fileNames.keys()):
    
            fname = fileNames[fileIndex]

            print "opening",fname, fileIndex,"/",len(fileNames),
            thisData = np.load(fname)
        
            print len(thisData['y']),"photons"

            if allData == None:

                allData = {}

                allData['sample'] = makeSampleId(fname, len(thisData['y']))
        
                for key in thisData.keys():

                    value = thisData[key]

                    if key != 'genDR':
                        if key == 'X':
                            # we only support sparse format here
                            addSparseRecHits(allData, thisData)
                        elif key == 'tracks':
                            addTracks(allData, thisData)
                        else:
                            # just copy the data 
                            allData[key] = value

                    # if not genDR
        
                # end loop over all items in the dict
        
            else:
                # append to existing data
          
                allData['sample'] = catItem(allData['sample'],
                                            makeSampleId(fname, len(thisData['y'])))

                for key in thisData.keys():
                    value = thisData[key]

                    if key != 'genDR':
                        if key == 'X':
                            # we only support sparse format here
                            addSparseRecHits(allData, thisData)
                        elif key == 'tracks':
                            addTracks(allData, thisData)
                        else:
                            # normal concatenation
                            allData[key] = catItem(allData[key], thisData[key])
                        
                    # end if not genDR
                
                # end loop over all items in the dict
            # end if not first
      
        # end of loop over file names for this base name
      
        # write out
        print "writing",outputFname,"(",len(allData['y']),"photons )"

        # it looks like **kwds is able to deal with slashes in the key names of the dict...
        np.savez(outputFname, **allData)
  
    # end of looping over base names (processes)

# end of loop over barrel/endcap

