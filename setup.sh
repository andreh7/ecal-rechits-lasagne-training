#!/usr/bin/env python

# export PYTHONHOME=~/python
# export PATH=$PYTHONHOME/bin:$PATH
# export LD_LIBRARY_PATH=$PYTHONHOME/lib:$LD_LIBRARY_PATH

source ~/virtualenv/py27-cuda75/bin/activate

CUDA_VERSION=7.5
#----------
# cudnn
#----------

# see http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html
CUDNN_DIR=~/cudnn-${CUDA_VERSION}

export LD_LIBRARY_PATH=$CUDNN_DIR/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDNN_DIR/cuda/include:$CPATH
# note the missing LD_
export LIBRARY_PATH=$CUDNN_DIR/cuda/lib64:$LD_LIBRARY_PATH
#----------

export PYTHONPATH=~/torchio:$PYTHONPATH

# to avoid picking up /usr/lib64/R/lib/libopenblas.so.0
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
