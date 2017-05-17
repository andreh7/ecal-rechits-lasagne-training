#!/usr/bin/env python
import os

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


parser.add_argument('dirs',
                    metavar = "dir",
                    type = str,
                    nargs = "+",
                    help='directories to make plots for',
                    )

options = parser.parse_args()
#----------------------------------------

if options.legendLoc != None and options.tmva:
    print >> sys.stderr,"--legend-loc is not supported with --tmva"
    sys.exit(1)
    

for theDir in options.dirs:

    cmdParts = []

    if options.tmva:
        cmdParts.append("./plotROCs-tmva.py")
    else:
        cmdParts.append("./plotROCs.py")
        cmdParts.append("--both")

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
                "0",
                "&"])

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

