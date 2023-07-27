FROM nvidia/cuda:10.0-runtime  as intermediate

ENV PYTHON_VERSION="3.6.5"

# install git
RUN apt-get update
RUN apt-get install -y git


RUN apt update && apt install -y --no-install-recommends \
    git \
    wget \
    unzip \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-opencv

RUN apt update && apt install -y unzip

RUN wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh

RUN chmod +x bazel-0.26.1-installer-linux-x86_64.sh \
    && ./bazel-0.26.1-installer-linux-x86_64.sh


RUN pip3 install --upgrade pip
RUN pip3 install numpy \
    tensorflow-gpu==1.14.0 \
    opencv-python==4.1.1.26 \
    keras==2.2.5 \
    Pillow \
    imageio \
    sklearn \
    scikit-learn \
    matplotlib \
    cryptography \
    segmentation_models \
    pytest-shutil \
    community

RUN pip3 install opencv-python
RUN pip3 uninstall -y tensorboard

