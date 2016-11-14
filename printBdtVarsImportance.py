#!/usr/bin/env python

# analyses a bdt variable importance run and prints some numbers

import sys, re
from pprint import pprint


#----------------------------------------------------------------------

def readFromLogFiles(logFnames):
    # look for messages like:
    #   test AUC when removing s4 (11 variables remaining) : 0.943595981598


    stepData = []

    thisStep = None
    prevNumRemainingVars = None

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
                        stepData.append(thisStep)

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
        stepData.append(thisStep)

    return stepData


#----------------------------------------------------------------------


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
