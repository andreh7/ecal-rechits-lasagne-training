#!/usr/bin/env python

export PYTHONHOME=~/python
export PATH=$PYTHONHOME/bin:$PATH
export LD_LIBRARY_PATH=$PYTHONHOME/lib:$LD_LIBRARY_PATH

# see http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html
CUDNN_DIR=~/cudnn-7.5

export LD_LIBRARY_PATH=$CUDNN_DIR/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDNN_DIR/cuda/include:$CPATH
export LIBRARY_PATH=$CUDNN_DIR/cuda/lib64:$LD_LIBRARY_PATH
