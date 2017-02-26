#!/usr/bin/env python

# splits a rechits file into test and train file

import sys
import numpy as np
import fnmatch

#----------------------------------------------------------------------

# fraction of test sample
testFraction = 0.25

randSeed = 1337

#----------------------------------------------------------------------

# @param ignoreKeys is a list of fnmatch patterns
def makeOutputVar(inputData, indices, ignoreKeys):
    # inputData must be a dict
  
    outputData = {}
  
    for key, value in inputData.items():
      
        # check if we should not touch this key
        if any(fnmatch.fnmatch(key, pattern) for pattern in ignoreKeys):
            continue

        # note that in the npy version value is never a dict 
        # but a numpy object

        outputData[key] = value[indices]

    # end of loop over items in the table
  
    return outputData

#----------------------------------------------------------------------

def makeOutputDataRecHits(indices, inputData, outputData):

    numOutputRows = len(indices)
  
    #----------
    # calculate the total number of output rechits
    #----------
    numOutputRecHits = inputData['X/numRecHits'][indices].sum()
  
    #----------
  
    # now that we know the total number of output rechits, 
    # copy the vectors related to the rechits 

    outputData['X/x']      = -1 * np.ones(numOutputRecHits, dtype = 'int32')
    outputData['X/y']      = -1 * np.ones(numOutputRecHits, dtype = 'int32')
    outputData['X/energy'] = -1 * np.ones(numOutputRecHits, dtype = 'float32')
  
    outputData['X/firstIndex'] = -1 * np.ones(numOutputRows, dtype = 'int32')
    outputData['X/numRecHits'] = inputData['X/numRecHits'][indices]
  
    # note that we keep the one based convention from Torch here for
    # historical reasons
    firstIndex = 1

    import tqdm
    progbar = tqdm.tqdm(total = numOutputRows, 
                        mininterval = 0.1, 
                        unit = 'photons',
                        desc = 'splitting rechits')
  

    # assign to variables to avoid dict lookups all the time (which makes it very slow, at least for npz files...)

    inputFirstIndex = inputData['X/firstIndex']
    inputNumRecHits = inputData['X/numRecHits']
    inputDataX = inputData['X/x']     
    inputDataY = inputData['X/y']     
    inputDataE = inputData['X/energy']

    outputDataX = outputData['X/x']
    outputDataY = outputData['X/y']
    outputDataE = outputData['X/energy']
    outputDataFirstIndex = outputData['X/firstIndex']
    outputDataNumRecHits = outputData['X/numRecHits']

    for i in range(numOutputRows):

        # this is zero based
        index = indices[i]
    
        outputDataFirstIndex[i] = firstIndex
    
        # sanity check of input data
        assert inputFirstIndex[index] >= 1
        assert inputFirstIndex[index] + inputNumRecHits[index] - 1 <= len(inputDataE), "failed at index=" + str(index)
    
        # baseInputIndex is zero based
        baseInputIndex = inputFirstIndex[index] - 1

        thisNumRecHits = inputNumRecHits[index]
        outputDataNumRecHits[i] = thisNumRecHits

        # copy coordinates and energy over
        # note that firstIndex is one based
        

        outputDataX[(firstIndex - 1):(firstIndex - 1 + thisNumRecHits)] = inputDataX[(baseInputIndex):(baseInputIndex + thisNumRecHits)] 
        outputDataY[(firstIndex - 1):(firstIndex - 1 + thisNumRecHits)] = inputDataY[(baseInputIndex):(baseInputIndex + thisNumRecHits)] 
        outputDataE[(firstIndex - 1):(firstIndex - 1 + thisNumRecHits)] = inputDataE[(baseInputIndex):(baseInputIndex + thisNumRecHits)] 
      
        firstIndex += thisNumRecHits
      
        # end -- loop over rechits of this photon
  
        progbar.update(1)

    # end -- loop over photons

    progbar.close()

#----------------------------------------------------------------------

def makeOutputDataTracks(indices, inputData, outputData):

    numOutputRows = len(indices)
  
    #----------
    # calculate the total number of output tracks
    #----------
    numOutputTracks = inputData['tracks/numTracks'][indices].sum()
  
    #----------
  
    # now that we know the total number of output rechits, 
    # copy the vectors related to the rechits 
  
    # variables other than the indexing variables 'firstIndex'
    # and 'numTracks'
    otherVarNames = []
  
    for key, value in inputData.keys():
        if not key.startswith("tracks/"):
            continue

        if key == 'tracks/firstIndex' or key == 'tracks/numTracks':
            outputData['key'] = -1 * np.ones(numOutputRows, dtype = 'int32')
    
        elif key == 'tracks/charge':
            # normal int variables
            outputData[key] = -1 * np.ones(numOutputTracks, dtype = 'int32')
            otherVarNames.append(key)
    
        else:
          # assume it's a float vector
          outputData[key] = -1 * np.ones(numOutputTracks, dtype = 'float32')
    
          otherVarNames.append(key)

    # end -- loop over keys of inputData.tracks

    # note that we keep the Torch/Lua convention of one based indices
    firstIndex = 1
  
    # make local variables for input and output variables
    # to avoid dict/npz file lookups within the loop
    inputDataNumTracks   = inputData['tracks/numTracks']
    inputDataFirstIndex  = inputData['tracks/firstIndex']
    inputDataRelpt       = inputData['tracks/relpt']

    outputDataFirstIndex = outputData['tracks/firstIndex']
    

    for i in range(numOutputRows):
  
        index = indices[i]
    
        outputDataFirstIndex[i] = firstIndex
        outputDataNumTracks[i]  = inputDataNumTracks[index]
    
        # sanity check of input data
        assert inputDataFirstIndex[index] >= 1
        assert inputDataFirstIndex[index] + inputDataNumTracks[index] - 1 <= len(inputDataRelpt), "failed at index=" + str(index)
    
        baseInputIndex = inputDatafirstIndex[index] - 1
    
        for j in range(inputDataNumTracks)[index]:
            # copy per track variables over
      
            for varname in otherVarNames:
                outputData[varname][firstIndex] = inputData.tracks[varname][baseInputIndex + j] 
      
            firstIndex += 1
    
        # end -- loop over tracks of this photon
  
    # end -- loop over photons

#----------------------------------------------------------------------

def makeOutputData(indices, inputData):

    # copy everything except the rechits
    outputData = makeOutputVar(inputData, indices, [ 'X/*', 'tracks/*' ])
  
    makeOutputDataRecHits(indices, inputData, outputData)
    # makeOutputDataTracks(indices, inputData, outputData)
  
    return outputData

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

ARGV = sys.argv[1:]
assert len(ARGV) == 1

np.random.seed(randSeed)

inputFile = ARGV.pop(0)

#----------
# generate output file test names
#----------
if inputFile.endswith(".npz"):
    outputFilePrefix = inputFile[:-4]
else:
    outputFilePrefix = inputFile

outputFileTrain = outputFilePrefix + "-train.npz"
outputFileTest  = outputFilePrefix + "-test.npz"

#----------

inputData = np.load(inputFile)
numEvents = len(inputData['y'])
print "have", numEvents,"photons"


# throw a random number for each event 
randVals = np.random.rand(numEvents)
trainIndices = np.arange(numEvents)[np.where(randVals > testFraction)]
testIndices = np.arange(numEvents)[np.where(randVals <= testFraction)]

assert len(trainIndices) + len(testIndices) == numEvents

#----------
# create and fill the output tensors
#----------

print "filling train dataset"
outputDataTrain = makeOutputData(trainIndices, inputData)

print "writing train dataset (",len(trainIndices),"photons) to",outputFileTrain

# note that this is uncompressed
np.savez(outputFileTrain, **outputDataTrain)


print "filling test dataset"
outputDataTest = makeOutputData(testIndices, inputData)

print "writing test dataset (", len(testIndices),"photons) to",outputFileTest
np.savez(outputFileTest, **outputDataTest)