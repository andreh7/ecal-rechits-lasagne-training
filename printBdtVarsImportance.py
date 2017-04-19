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

def printStepDataToCSV(stepData, os = sys.stdout):
    # prints the information about the removed variables
    # in .csv format

    print >> os, ",".join([
            "numRemainingVars",
            "removedVar",
            "auc"])

    for line in stepData:
        for removedVar, auc in line['aucWithVarRemoved'].items():
            parts = [
                line['numRemainingVars'],
                removedVar,
                auc
                ]
            print >> os, ",".join([ str(x) for x in parts ])



#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

if __name__ == '__main__':

    import bdtvarsimportanceutils

    ARGV = sys.argv[1:]

    assert len(ARGV) >= 1

    # find complete directories
    completeDirs, incompleteDirs = bdtvarsimportanceutils.findComplete(ARGV[0])

    print "complete directories:"
    for theDir in sorted(completeDirs.values()):
        print "  %s" % theDir

    print "incomplete directories:"
    for theDir in sorted(incompleteDirs.values()):
        print "  %s" % theDir



    # stepData = readFromLogFiles(ARGV)
    stepData, fullNetworkAUC = bdtvarsimportanceutils.readFromTrainingDir(ARGV[0])

# pprint(stepData)
# printStepDataToCSV(stepData)
# sys.exit(1)
    
print "order of removal:"
print "%-30s: %.4f" % ('before', fullNetworkAUC)
for step in stepData:
    print "%-30s: %.4f" % (step['removedVariable'],step['aucWithVarRemoved'][step['removedVariable']])

#----------
# make plots
#----------

import pylab
pylab.figure(facecolor='white', figsize = (20,12))

xvalues = []
yvalues = []
for step in stepData:
    for varname, auc in step['aucWithVarRemoved'].items():
        xvalues.append(step['numRemainingVars'])
        yvalues.append(auc)

pylab.plot(xvalues, yvalues, 'o')


if True:
    # add labels
    pylab.ylim((0.6, pylab.ylim()[1]))

    ystart = 0.75
    ylineHeight = 0.01

    for step in stepData:
        xpos = step['numRemainingVars']

        ypos = ystart

        for index, (auc, varname) in enumerate(sorted([ (auc, varname) for varname, auc in step['aucWithVarRemoved'].items() ],
                                   reverse = False)):
            # lowest AUC first

            label = varname

            if index == step['numRemainingVars']:
                color = 'red'
            else:
                color = 'black'

            pylab.text(
                xpos,
                ypos,
                label, 
                ha = 'center', va = 'top',
                fontsize = 10,
                color = color,
                )

            ypos += ylineHeight




pylab.grid()
pylab.xlabel('number of remaining input variables')
pylab.ylabel('test auc when removing variable')

pylab.savefig("bdt-vars-importance.png")
pylab.savefig("bdt-vars-importance.pdf")

pylab.show()
