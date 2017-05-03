#!/usr/bin/env python
import sys, os


# see the list of theano flags here:
#   http://deeplearning.net/software/theano/library/config.html#config-attributes

# OPENBLAS_NUM_THREADS=1 THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,blas.ldflags=-L/home/users/aholz/openblas-0.2.15/lib\ -lopenblaso python $@

# 'FAST_RUN': Apply all optimizations, and use C implementations where possible
#             (which is also the default mode)


# parse command line arguments
import argparse

parser = argparse.ArgumentParser(prog='run-gpu.py',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                 )


parser.add_argument('--gpu',
                    default = "1",
                    type = str,
                    help='gpu to run on or the string cpu',
                    )

parser.add_argument('--memfrac',
                    default = None,
                    type = float,
                    help = "memory fraction to allocate for theano"
                    )

parser.add_argument('args',
                    metavar = "args",
                    type = str,
                    nargs = '*',
                    )

options = parser.parse_args()

cmdParts = []

if options.gpu == 'cpu':
    
    flags = [
        "mode=FAST_RUN",
        "device=cpu",
        "floatX=float32",
        "exception_verbosity=high",
        "optimizer=None",
        ]

else:
    # assume a gpu number
    options.gpu = int(options.gpu)
    #----------
    # theano flags
    #----------
    # old style, does NOT require pygpu (which I can't get to work)
    deviceName = "gpu%d"  % options.gpu

    # new style, requires pygpu which I can't get to work
    # deviceName = "cuda%d" % options.gpu

    flags = [
        "mode=FAST_RUN",
        "device=%s" % deviceName,
        "floatX=float32",
        "dnn.enabled=True",
        ]

    if options.memfrac != None:
        flags.append("lib.cnmem=%f"  % options.memfrac)

cmdParts.append("THEANO_FLAGS=" + ",".join(flags))
#----------

cmdParts.append("python")

cmdParts.extend(options.args)

# print "cmd="," ".join(cmdParts)
res = os.system(" ".join(cmdParts))
sys.exit(res)
