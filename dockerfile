FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    vim \
    wget \
    unzip \
    yasm \
    pkg-config \
    curl \
    python3-dev \
    python3-pip \
    python3-testresources 


RUN pip3 install pip -U
RUN pip3 install -U setuptools pip
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install scikit-learn
RUN pip3 install tqdm
RUN pip3 install scipy
RUN git clone git@github.com:remokasu/somkit.git
RUN cd somkit && pip3 install -e .