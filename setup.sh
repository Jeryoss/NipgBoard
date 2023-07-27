#!/bin/bash

chmod +x design.sh
chmod +x run.sh
wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh
chmod +x bazel-0.26.1-installer-linux-x86_64.sh
./bazel-0.26.1-installer-linux-x86_64.sh --user
pip install --user virtualenv
KIRA_PYLOC="$(which python3)"
virtualenv kira_venv -p $KIRA_PYLOC
source kira_venv/bin/activate
pip3 install --upgrade pip
pip3 install --requirement requirements.txt
pip3 uninstall -y tensorboard
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export PYTHONPATH=$PYTHONPATH:${PWD}
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDADIR=/usr/local/cuda-10.0
export CUDA_HOME=/usr/local/cuda-10.0
