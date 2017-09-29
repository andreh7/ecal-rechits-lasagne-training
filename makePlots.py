#!/usr/bin/env python
import os, sys

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

# parse command line arguments
import argparse

parser = argparse.ArgumentParser(prog='drawSparseRecHits',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                 )


parser.add_argument('--legend-loc',
                    metavar = "loc",
                    type = str,
                    default = None,
                    help='legend location specification',
                    dest = "legendLoc",
                    )


parser.add_argument('--nodate',
                    default = False,
                    action = 'store_true',
                    help='suppress date label',
                    )

parser.add_argument('--tmva',
                    default = False,
                    action = 'store_true',
                    help='make plots for TMVA trainings',
                    )

parser.add_argument("--min-epoch",
                    dest = 'minEpoch',
                    type = int,
                    default = None,
                    help="first epoch to plot (useful e.g. if the training was far off at the beginning)",
                    )

parser.add_argument("--max-epoch",
                    dest = 'maxEpoch',
                    type = int,
                    default = None,
                    help="last epoch to plot (useful e.g. if the training diverges at some point)",
                    )

parser.add_argument('dirs',
                    metavar = "dir",
                    type = str,
                    nargs = "+",
                    help='directories to make plots for',
                    )

options = parser.parse_args()
#----------------------------------------

if options.tmva:
    if not options.legendLoc is None:
        print >> sys.stderr,"--legend-loc is not supported with --tmva"
        sys.exit(1)
    
    if not options.minEpoch is None:
        print >> sys.stderr,"--min-epoch is not supported with --tmva"
        sys.exit(1)

    if not options.maxEpoch is None:
        print >> sys.stderr,"--max-epoch is not supported with --tmva"
        sys.exit(1)
        

for theDir in options.dirs:

    cmdParts = []

    if options.tmva:
        cmdParts.append("./plotROCsTMVA.py")
    else:
        cmdParts.append("./plotROCs.py")
        cmdParts.append("--both")

    if not options.minEpoch is None:
        cmdParts.append("--min-epoch " + str(options.minEpoch))

    if not options.maxEpoch is None:
        cmdParts.append("--max-epoch " + str(options.maxEpoch))

    cmdParts.extend([
            "--save-plots",
            theDir
            ])

    if options.nodate:
        cmdParts.append("--nodate")

    if options.legendLoc != None:

        validLegendLocs = [
            "right",
            "center left",
            "upper right",
            "lower right",
            "best",
            "center",
            "lower left",
            "center right",
            "upper left",
            "upper center",
            "lower center",
            ]

        if not options.legendLoc in validLegendLocs:
            print >> sys.stderr,"unsupported legend location '%s'. Supported are: %s" % (
                options.legendLoc,
                ", ".join(validLegendLocs)
                )
            sys.exit(1)

        cmdParts.append("--legend-loc '" + options.legendLoc + "'")

    cmdParts.append("&")

    if not options.tmva:
        cmdParts.extend([
                "./plotNNoutput.py",
                "--save-plots",
                "--sample train",
                theDir,
                ])

        if options.maxEpoch is None:
            cmdParts.append("0")
        else:
            cmdParts.append(str(options.maxEpoch))

        cmdParts.append("&")

    # wait for the two previous processes to finish
    cmdParts.extend([ "wait",
                      ";"])
    
    cmdParts.extend([
            "mailme",
            "--subject " + theDir,
            theDir + "/*.pdf" ])

    cmd = " ( " + " ".join(cmdParts) + " ) &"

    # print cmd
    os.system(cmd)

