FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
RUN apt update

# Install common build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    curl \
    dialog apt-utils \
    git \
    g++ \
    libboost-all-dev \
    libssl-dev \
    libtool \
    libyaml-cpp-dev \
    lsb-release \
    make \
    pkg-config \
    python3-pip \
    python3-setuptools \
    wget \
    cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# update pip & install cmake for tensorrt build
RUN python3 -m pip install --upgrade pip && pip3 install cmake==3.18.0


# install debian pkgs (should be added above)
RUN apt update
RUN apt install -y libpcap-dev \
                   libjson-c-dev \
                   libjsoncpp-dev \
                   psmisc \
                   net-tools \
                   libglfw3-dev libglew-dev libeigen3-dev libtclap-dev \
                   libproj-dev \
                   ffmpeg \
                   wmctrl \
                   xdotool \
                   iputils-ping \
                   python \
                   libcgal-dev

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


WORKDIR /home/CUDA/