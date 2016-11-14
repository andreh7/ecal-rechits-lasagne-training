#!/usr/bin/env python

# analyses a bdt variable importance run and prints some numbers

import sys, re, os
from pprint import pprint


#----------------------------------------------------------------------

def readFromLogFiles(logFnames):
    # reconstruct the AUC values from the log files
    # (for older runs where the detailed information was not written
    # out into separate files)

    # look for messages like:
    #   test AUC when removing s4 (11 variables remaining) : 0.943595981598

    stepData = []

    thisStep = None
    prevNumRemainingVars = None

    #----------

    def completeStep(thisStep):
        # adds additional information
        stepData.append(thisStep)

        # get the list of all variables
        assert len(thisStep['aucWithVarRemoved']) == thisStep['numRemainingVars'] + 1, "numRemainingVars is %d, number of variable removed entries is %d" % (
            len(thisStep['aucWithVarRemoved']), thisStep['numRemainingVars'])

        # get the list of all variables in this step
        thisStep['allVariables'] = thisStep['aucWithVarRemoved'].keys()

        # best AUC after removal
        thisStep['bestAUC'], varToRemove = max( [ (auc, var) for var, auc in thisStep['aucWithVarRemoved'].items() ] )

        assert varToRemove == thisStep['removedVariable']

    #----------

    for logFile in logFnames:

        for line in open(logFile).read().splitlines():
            mo = re.match("test AUC when removing (\S+) \((\d+) variables remaining\) : (\S+)\s*$", line)

            if mo:

                removedVar = mo.group(1)
                numRemainingVars = int(mo.group(2))
                thisAUC = float(mo.group(3))

                if numRemainingVars != prevNumRemainingVars:
                    assert prevNumRemainingVars == None or numRemainingVars + 1 == prevNumRemainingVars
                    if thisStep != None:
                        completeStep(thisStep)

                    thisStep = dict(
                        numRemainingVars = numRemainingVars,
                        aucWithVarRemoved = {}
                        )

                    prevNumRemainingVars = numRemainingVars

                thisStep['aucWithVarRemoved'][removedVar] = thisAUC

                continue

            #----------
            mo = re.match("removing variable (\S+)\s*$", line)

            if mo:
                thisStep['removedVariable'] = mo.group(1)
                continue

            #----------

        # end of loop over lines
    # end of loop over input files

    if thisStep != None:
        completeStep(thisStep)

    return stepData


#----------------------------------------------------------------------

from bdtVarsImportance import getMeanTestAUC

def readVars(dirname):
    fin = open(os.path.join(dirname,"variables.py"))
    retval = eval(fin.read())
    fin.close()
    return retval

def readFromTrainingDir(trainDir):
    # reads from the training directory

    # read initial training
    overallDir = os.path.join(trainDir, "00-00")
    overallAUC = getMeanTestAUC(overallDir)
    allVars = readVars(overallDir)

    import itertools

    stepData = []

    allVarsCurrentStep = list(allVars)

    for index in itertools.count(1):
        thisStep = dict(
            aucWithVarRemoved = {},
            )

        remainingVars = None
        allVariablesOrdered = []

        for subIndex in itertools.count(1):
            # read variables of this step
            
            inputDir = os.path.join(trainDir,
                                    "%02d-%02d" % (index, subIndex))

            if not os.path.exists(inputDir):
                break

            thisVars = readVars(inputDir)

            if remainingVars == None:
                remainingVars = len(thisVars)
                thisStep['numRemainingVars'] = remainingVars
            else:
                assert remainingVars == len(thisVars)

            #----------
            # find which variable was removed
            #----------
            assert len(thisVars) + 1 == len(allVarsCurrentStep)

            removedVar = None
            
            for var in allVarsCurrentStep:
                if not var in thisVars:
                    removedVar = var
                    break

            assert removedVar != None

            allVariablesOrdered.append(removedVar)

            # read AUC
            thisAUC = getMeanTestAUC(inputDir)
            
            thisStep['aucWithVarRemoved'][removedVar] = thisAUC


        # end of loop over subindices

        if subIndex == 1:
            # no directory for this index found
            break

        # end of loop over subIndex

        #----------
        # complete some more information
        #----------
        thisStep['bestAUC'], varToRemove = max( [ (auc, var) for var, auc in thisStep['aucWithVarRemoved'].items() ] )
        thisStep['removedVariable'] = varToRemove
        thisStep['allVariables'] = allVariablesOrdered

        stepData.append(thisStep)

        # prepare next iteration
        allVarsCurrentStep.remove(varToRemove)

    # end of loop over indices

    return stepData


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

if __name__ == '__main__':

    ARGV = sys.argv[1:]

    assert len(ARGV) >= 1

    stepData = readFromLogFiles(ARGV)


pprint(stepData)
sys.exit(1)
    
print "order of removal:"
for step in stepData:
    print step['removedVariable'],step['aucWithVarRemoved'][step['removedVariable']]

#----------
# make plots
#----------

import pylab
pylab.figure(facecolor='white')

xvalues = []
yvalues = []
labels  = []
for step in stepData:
    for varname, auc in step['aucWithVarRemoved'].items():
        labels.append(varname)
        xvalues.append(step['numRemainingVars'])
        yvalues.append(auc)

pylab.plot(xvalues, yvalues, 'o')


# add labels
# from http://stackoverflow.com/a/5147430/288875
for x, y, label in zip(xvalues, yvalues, labels):
    pylab.annotate(
        label, 
        xy = (x, y), 
        xytext = (-20, 20),
        textcoords = 'offset points', 
        ha = 'right', va = 'bottom',
        # bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
        )



pylab.grid()
pylab.xlabel('number of remaining input variables')
pylab.ylabel('test auc')
pylab.show()
