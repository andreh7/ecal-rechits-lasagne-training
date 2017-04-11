#!/usr/bin/env python

# finds events in input files (npy only so far) 
# e.g. given the mvaid

import numpy as np

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

# parse command line arguments
import argparse

parser = argparse.ArgumentParser(prog='train01.py',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                 )

parser.add_argument('--mvaid',
                    dest = "mvaid",
                    type = float,
                    default = None,
                    help='filter events by mvaid'
                    )

parser.add_argument('inputFiles',
                    metavar = "input.npz",
                    type = str,
                    nargs = '+',
                    help='input data files to search',
                    )

options = parser.parse_args()

if not options.mvaid:
    print >> sys.stderr,"must specify at least one of",", ".join([
            "--mvaid",
            ])
    sys.exit(1)

#----------------------------------------

for fname in options.inputFiles:

    isTorch = fname.endswith(".t7")
    
    if isTorch:
        import torchio
        data = torchio.read(fname)
    else:
        data = np.load(fname)

    # produce a vector of bools
    indices = np.ones(len(data['y']), dtype = np.bool)

    if options.mvaid:
        mvaid = data['mvaid']
        if isTorch:
            mvaid = mvaid.asndarray()

        indices = indices & ( np.abs(mvaid - options.mvaid) < 1e-8)
    
    if not np.any(indices):
        # encourage garbage collection
        del data
        continue

    for ind in np.where(indices)[0]:
        print "%s:%d" % (fname, ind),

        if options.mvaid:
            print "mvaid=" + str(mvaid[ind])

    del data


