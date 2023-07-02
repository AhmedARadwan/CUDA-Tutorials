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

# Install ROS noetic on Ubuntu (http://wiki.ros.org/noetic/Installation/Ubuntu)
ENV DEBIAN_FRONTEND noninteractive
ENV ROS_DISTRO=melodic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update && apt-get install -y ros-${ROS_DISTRO}-desktop ros-${ROS_DISTRO}-velodyne-pointcloud
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN apt update && apt install -y python3-catkin-tools \
                                 python-rosdep \
                                 python-rosinstall \
                                 python-rosinstall-generator \
                                 python-wstool \
                                 build-essential

# install debian pkgs
RUN apt install -y ros-${ROS_DISTRO}-ros-numpy \
                   ros-${ROS_DISTRO}-jsk-recognition-msgs \
                   ros-${ROS_DISTRO}-autoware-msgs \
                   ros-${ROS_DISTRO}-cv-bridge \
                   ros-${ROS_DISTRO}-image-transport \
                   ros-${ROS_DISTRO}-camera-info-manager \
                   ros-${ROS_DISTRO}-tf2-sensor-msgs \
                   ros-${ROS_DISTRO}-nmea-msgs \
                   ros-${ROS_DISTRO}-diagnostic-updater \
                   ros-${ROS_DISTRO}-roslint \
                   ros-${ROS_DISTRO}-angles \
                   ros-${ROS_DISTRO}-pcl-ros \
                   ros-${ROS_DISTRO}-ros-numpy \
                   ros-${ROS_DISTRO}-tf-conversions \
                   ros-${ROS_DISTRO}-tf2-geometry-msgs \
                   ros-${ROS_DISTRO}-rviz \
                   ros-${ROS_DISTRO}-jsk-rviz-plugins



ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics


WORKDIR /home/CUDA/