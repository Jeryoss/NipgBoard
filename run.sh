#!/bin/bash

if [[ $1 == '' ]]; then
  	echo "No \"logdir\" data directory given! Please run script as './run.sh <logdir>'."
    echo "Make sure to add the absolute path to the logdir containing the data."
else
    export CUDA_VISIBLE_DEVICES=1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
    export PYTHONPATH=$PYTHONPATH:${PWD}
    export PATH=$PATH:/usr/local/cuda-10.0/bin
    export CUDADIR=/usr/local/cuda-10.0
    export CUDA_HOME=/usr/local/cuda-10.0
    if [[ $2 == '' ]]; then
        bazel run tensorboard -- --logdir=$1 --port=8900
    else
        bazel run tensorboard -- --logdir=$1 --port=$2
    fi
fi
