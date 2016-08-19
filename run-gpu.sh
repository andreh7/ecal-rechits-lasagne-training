#!/bin/sh


# see the list of theano flags here:
#   http://deeplearning.net/software/theano/library/config.html#config-attributes

# OPENBLAS_NUM_THREADS=1 THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,blas.ldflags=-L/home/users/aholz/openblas-0.2.15/lib\ -lopenblaso python $@

# 'FAST_RUN': Apply all optimizations, and use C implementations where possible
#             (which is also the default mode)

THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,dnn.enabled=True python $@