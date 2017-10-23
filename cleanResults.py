#!/usr/bin/env python

import sys, os, re, time, glob

#---------------------------------------------------------------------- 
# main
#---------------------------------------------------------------------- 

ARGV = sys.argv

from optparse import OptionParser
parser = OptionParser("""

  usage: %prog [options] dir1 [ dir2 ]

  script to remove the following from a results directory:
 
   - network model files older than minimum age and before the most recent one
  
   - test/train output data older minimum age and before the most recent
     one and only if a corresponding cached AUC file exists

     will also delete compressed files

"""
)

parser.add_option("-n",
                  dest="dryRun",
                  default = False,
                  action="store_true",
                  help="only print what would be deleted but do not actually delete the files",
                  )


parser.add_option("--min-age",
                  dest="minAge",
                  default = 5.0,
                  type = float,
                  help="minimum age in minutes below which a file will not be deleted. Floating point values are accepted.",
                  )

parser.add_option("--keep-epoch",
                  dest="keepEpochList",
                  default = None,
                  type = str,
                  help="comma separated list of epoch numbers to keep anyway",
                  )

parser.add_option("--keep",
                  dest="numFilesToKeep",
                  default = 1,
                  type = int,
                  help="number of last iterations to keep. default is 1, i.e. keep only the last one",
                  metavar = "n"
                  )

(options, ARGV) = parser.parse_args()

if len(ARGV) < 1:
    print >> sys.stderr,"must specify at least one directory to work on"
    sys.exit(1)

# convert minimum age to seconds
options.minAge *= 60.0

if options.keepEpochList != None:
    options.keepEpochList = [ int(x) for x in options.keepEpochList.split(',') ]

#----------------------------------------

for dirname in ARGV:

    allFnames = set(os.listdir(dirname))

    # list of  (number, full filename) 
    modelFiles = []

    # lists of (number, full filename)
    outputFiles = dict(train = [], test = [])

    # mostly for verbose printing
    filesToDelete = []
    filesToKeep = []
    filesTooYoung = [] # files which would be deleted but fail the minimum age requirement
    unknownFiles = []

    # go through all files in this directory
    for fname in allFnames:

        fullFname = os.path.join(dirname, fname)
        #----------
        # training output per event files
        #----------

        if fname in ("roc-data-test-mva.t7", "roc-data-train-mva.t7",
                     "roc-data-test-mva.npz", "roc-data-train-mva.npz",
                     ):
            filesToKeep.append(fullFname)
            continue

        mo = re.match("roc-data-(train|test)-(\d+)\.t7$", fname)
        if not mo:
            mo = re.match("roc-data-(train|test)-(\d+)\.npz$", fname)
        if mo:
            # add it to the list
            sample = mo.group(1)
            index = int(mo.group(2), 10)

            outputFiles[sample].append( (index, fullFname) )
            continue

        # AUC cached files
        if fname.endswith(".t7.cached-auc.py") or fname.endswith(".npz.cached-auc.py"):
            filesToKeep.append(fullFname)
            continue

        #----------
        # model files
        #----------
        mo = re.match("model(\d+)\.net$", fname)

        if not mo:
            mo = re.match("model-(\d+)\.npz$", fname)

        if not mo:
            mo = re.match("checkpoint-(\d+)\.torch$", fname)

        if mo:
            modelNumber = int(mo.group(1), 10)

            modelFiles.append( (modelNumber, fullFname) )

            continue

        unknownFiles.append(fullFname)

    # end of loop over files

    #----------
    # keep only latest few model file
    #----------

    if modelFiles:
        modelFiles.sort(key = lambda line: line[0])

        # only keep the last one
        filesToKeep.extend([ theFile[1] for theFile in modelFiles[-options.numFilesToKeep:] ])

        for line in modelFiles[:-options.numFilesToKeep]:

            epoch, fname = line

            if options.keepEpochList != None and epoch in options.keepEpochList:
                # keep this epoch anyway
                filesToKeep.append(fname)
            else:
                filesToDelete.append(fname)

    #----------
    # keep only output files for which there is a cached
    # version and the latest one
    #----------
    for sample in ('train', 'test'):
        lines = outputFiles[sample]

        if not lines:
            print >> sys.stderr,"WARNING: no %s files found" % sample
            continue

        lines.sort(key = lambda line: line[0])
            
        # keep the latest one
        filesToKeep.extend([ line[1] for line in lines[-options.numFilesToKeep:] ])

        # keep previous ones if there is no cached file
        for line in lines[:-options.numFilesToKeep]:
            epoch, fname = line

            keep = False

            if options.keepEpochList != None and epoch in options.keepEpochList:
                # keep this epoch anyway
                keep = True
            
            cachedFname = fname + ".cached-auc.py"
            if not cachedFname in filesToKeep:
                # we must keep this file, there is no cached version
                keep = True

            if keep:
                # we need to keep this file
                filesToKeep.append(fname)
            else:
                filesToDelete.append(fname)

    #----------
    # apply minimum age criterion to candidates to be deleted
    #----------
    now = time.time()
    for i in reversed(range(len(filesToDelete))):
        
        timestamp = os.path.getmtime(filesToDelete[i])

        age = now - timestamp 

        if age < options.minAge:
            # file is too new
            filesTooYoung.append(filesToDelete[i])
            del filesToDelete[i]

    # end of loop over files to delete

    if options.dryRun:
        for fileList, description in [
            (filesToDelete, "files which would be deleted:"),
            (filesTooYoung, "files which would be deleted but are too young:"),
            (filesToKeep,   "files which would be kept"),
            (unknownFiles,  "unknown types of files:"),
            ]:

            if fileList:
                fileList.sort()
                print description
                for fname in fileList:
                    print "  ",fname

    else:
        # not dryrun
        for fname in filesToDelete:
            print "deleting",fname
            os.unlink(fname)

    # zip any remaining .npz files
    fnames = glob.glob(dirname + "/*.npz")
    for fname in fnames:
        if options.dryRun:
            if not fname in filesToDelete:
                print "would zip",fname
        else:
            print "zipping",fname
            os.system("bzip2 " + fname)

# end of loop over directories
