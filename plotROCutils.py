#!/usr/bin/env python

#----------------------------------------------------------------------

def addTimestamp(inputDir, x = 0.0, y = 1.07, ha = 'left', va = 'bottom'):

    import pylab, time, os

    # static variable
    if not hasattr(addTimestamp, 'text'):
        # make all timestamps the same during one invocation of this script

        now = time.time()

        addTimestamp.text = time.strftime("%a %d %b %Y %H:%M", time.localtime(now))

        # use the timestamp of the samples.txt file
        # as the starting point of the training
        # to determine the wall clock time elapsed
        # for the training

        fname = os.path.join(inputDir, "samples.txt")
        if os.path.exists(fname):
            startTime = os.path.getmtime(fname)
            deltaT = now - startTime

            addTimestamp.text += " (%.1f days)" % (deltaT / 86400.)


    pylab.gca().text(x, y, addTimestamp.text,
                     horizontalalignment = ha,
                     verticalalignment = va,
                     transform = pylab.gca().transAxes,
                     # color='green', 
                     fontsize = 10,
                     )
#----------------------------------------------------------------------

    
def addDirname(inputDir, x = 1.0, y = 1.07, ha = 'right', va = 'bottom'):

    import pylab

    if inputDir.endswith('/'):
        inputDir = inputDir[:-1]

    pylab.gca().text(x, y, inputDir,
                     horizontalalignment = ha,
                     verticalalignment = va,
                     transform = pylab.gca().transAxes,
                     # color='green', 
                     fontsize = 10,
                     )

#----------------------------------------------------------------------

def addNumEvents(numEventsTrain, numEventsTest):

    for numEvents, label, x0, halign in (
        (numEventsTrain, 'train', 0.00, 'left'),
        (numEventsTest, 'test',   1.00, 'right'),
        ):

        if numEvents != None:
            import pylab
            pylab.gca().text(x0, -0.08, '# ' + label + ' ev.: ' + str(numEvents),
                             horizontalalignment = halign,
                             verticalalignment = 'center',
                             transform = pylab.gca().transAxes,
                             fontsize = 10,
                             )

#----------------------------------------------------------------------

def readDescription(inputDir):
    import os

    descriptionFile = os.path.join(inputDir, "samples.txt")

    if os.path.exists(descriptionFile):

        description = []

        # assume that these are file names (of the training set)
        fnames = open(descriptionFile).read().splitlines()

        for fname in fnames:
            if not fname:
                continue

            fname = os.path.basename(fname)
            fname = os.path.splitext(fname)[0]

            if fname.endswith("-train"):
                fname = fname[:-6]
            elif fname.endswith("-test"):
                fname = fname[:-5]

            fname = fname.replace("_rechits","")

            description.append(fname)

        return ", ".join(description)

    else:
        return None

#----------------------------------------------------------------------
