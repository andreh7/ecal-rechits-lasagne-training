#!/usr/bin/env python

# analyses a bdt variable importance run and prints some numbers

import sys, re, os
from pprint import pprint

#----------------------------------------------------------------------

expectedNumEpochs = 200

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

    #----------
    # parse command line arguments
    #----------
    import argparse

    parser = argparse.ArgumentParser(prog='printBdtVarsImportance.py',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     )

    bdtvarsimportanceutils.fomAddOptions(parser)

    parser.add_argument('inputDir',
                        metavar = "inputDir",
                        type = str,
                        nargs = 1,
                        help='input directory with result files to read from',
                        )

    parser.add_argument("--save-plots",
                        dest = 'savePlots',
                        default = False,
                        action="store_true",
                        help="save plots in input directory",
                        )

    options = parser.parse_args()


    options.inputDir = options.inputDir[0]

    bdtvarsimportanceutils.fomGetSelectedFunction(options, expectedNumEpochs)

    #----------

    # find complete directories
    completeDirs, incompleteDirs = bdtvarsimportanceutils.findComplete(options.inputDir, expectedNumEpochs = expectedNumEpochs)

    print "complete directories:"
    for theDir in sorted(completeDirs.values()):
        print "  %s" % theDir

    print "incomplete directories:"
    for theDir in sorted(incompleteDirs.values()):
        print "  %s" % theDir



    # stepData = readFromLogFiles(ARGV)
    aucData = bdtvarsimportanceutils.readFromTrainingDir(options.inputDir, expectedNumEpochs = expectedNumEpochs, 
                                                         fomFunction = options.fomFunction,
                                                         numParallelProcesses = 10)

    aucData.removeVarnamePrefix('phoIdInput/')

    fullNetworkAUC = aucData.getOverallAUC()

    # from pprint import pprint
    # print pprint(aucData.data)

# pprint(stepData)
# printStepDataToCSV(stepData)
# sys.exit(1)
    
print "order of removal:"
print "%-30s: %.4f" % ('before', fullNetworkAUC)

# we only have to up to tot num vars minus two
for numVarsRemoved in range(aucData.getTotNumVars()):
    print "%2d vars removed:" % numVarsRemoved,

    # find the variable leading to the highest AUC
    # when removed
    varData = aucData.getStepAUCs(numVarsRemoved)

    if varData:
        worstVar = max([(step['aucWithVarRemoved'], step['removedVariable']) for step in varData ])[1]
    else:
        worstVar = None

    isStepComplete = aucData.isStepComplete(numVarsRemoved)

    if isStepComplete:
        print "complete"
    else:
        print "incomplete (%d results)" % aucData.getNumResultsAtStep(numVarsRemoved)

    for step in aucData.getStepAUCs(numVarsRemoved):
        print "%-30s: %.4f" % (step['removedVariable'],step['aucWithVarRemoved']),
        if step['removedVariable'] == worstVar:
            if isStepComplete:
                print "<<<",
            else:
                print "(<<<)",

        print

#----------
# make plots
#----------

import pylab
pylab.figure(facecolor='white', figsize = (20,12))

xvalues = []
yvalues = []
labels  = []
colors  = []
labels_ypos = []

# for variable name labels
ystart = 0.75
ylineHeight = 0.01

for numVarsRemoved in range(aucData.getTotNumVars()):
    numRemainingVars = aucData.getTotNumVars() - numVarsRemoved

    isCompleteStep = aucData.isStepComplete(numVarsRemoved)

    # if isCompleteStep:
    #     worstVar = aucData.worstVar(numVarsRemoved)
    # else:
    #     worstVar = None

    ypos = ystart

    # sort by lowest AUC first
    for index, step in enumerate(
                         sorted(aucData.getStepAUCs(numVarsRemoved),
                         key = lambda item: item['aucWithVarRemoved']
                         )):
        xvalues.append(numRemainingVars)
        yvalues.append(step['aucWithVarRemoved'])
        labels.append(step['removedVariable'])

        if isCompleteStep:
            # label the variable with highest AUC when removed in red
            if index == numRemainingVars:
                colors.append('red')
            else:
                colors.append('black')

        else:
            # mark variables of incomplete steps in gray
            colors.append('gray')

        labels_ypos.append(ypos)

        # prepare next iteration
        ypos += ylineHeight



# plot data points
pylab.plot(xvalues, yvalues, 'o')

# add labels
if True:

    pylab.ylim((0.6, pylab.ylim()[1]))

    for xpos, ypos, label, color in zip(xvalues, labels_ypos, labels, colors):

            pylab.text(
                xpos,
                ypos,
                label, 
                ha = 'center', va = 'top',
                fontsize = 10,
                color = color,
                )

pylab.grid()
pylab.xlabel('number of remaining input variables')
pylab.ylabel('test auc when removing variable')

if options.savePlots:
    for suffix in ("png", "pdf"):
        fname = os.path.join(options.inputDir, "bdt-vars-importance." + suffix)
        pylab.savefig(fname)
        print "wrote plot to",fname

else:
    pylab.show()
