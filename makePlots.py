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


parser.add_argument('dirs',
                    metavar = "dir",
                    type = str,
                    nargs = "+",
                    help='directories to make plots for',
                    )

options = parser.parse_args()
#----------------------------------------

for theDir in options.dirs:

    cmdParts = [
            "./plotROCs.py",
            "--save-plots",
            "--both",
            theDir
            ]

    if options.legendLoc != None:
        cmdParts.append("--legend-loc '" + options.legendLoc + "'")

    cmdParts.append("&")
    
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

