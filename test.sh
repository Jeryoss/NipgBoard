#!/bin/bash

if [[ $1 == '' ]]; then
  	echo "No \"logdir\" data directory given! Please run script as './run.sh <logdir>'."
    echo "Make sure to add the absolute path to the logdir containing the data."
else
    source kira_venv/bin/activate
    export CUDA_VISIBLE_DEVICES=1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
    export PYTHONPATH=$PYTHONPATH:${PWD}
    export PATH=$PATH:/usr/local/cuda-10.0/bin
    export CUDADIR=/usr/local/cuda-10.0
    export CUDA_HOME=/usr/local/cuda-10.0
    bazel run tensorboard -- --logdir=$1 --port=9009 & PIDIOS=$!
    python3 run_frontend_test.py & PIDMIX=$!
    wait $PIDMIX
fi
