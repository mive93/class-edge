FROM nvcr.io/nvidia/tensorrt:20.06-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git build-essential cmake rsync libeigen3-dev libglew-dev libglfw3-dev freeglut3-dev libfreetype6-dev libyaml-cpp-dev libpcap-dev libmatio-dev

RUN apt-get install -y python3-matplotlib python-dev libgdal-dev libcereal-dev python-numpy
RUN apt-get install -y libgles2-mesa-dev xorg-dev libglu1-mesa-dev
RUN apt-get install -y libopencv-dev libopencv-contrib-dev

WORKDIR /root

RUN mkdir repos

WORKDIR /root/repos

RUN apt-get install -y build-essential \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-venv \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libdc1394-22-dev \
    libavresample-dev

RUN git clone https://github.com/opencv/opencv.git && git clone https://github.com/opencv/opencv_contrib.git

RUN mkdir -p opencv/build

WORKDIR /root/repos/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH='/root/repos/opencv_contrib/modules' \
    -D BUILD_EXAMPLES=ON \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=7.0 \
    -D CUDA_ARCH_PTX="" \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_TBB=ON \
    ../

RUN make -j4
RUN make install
RUN ldconfig

#WORKDIR /root/repos/
#RUN git clone https://git.hipert.unimore.it/mverucchi/class-edge.git
ENV AAA=BBB
ADD . /root/repos/class-edge
WORKDIR /root/repos/class-edge
RUN wget https://github.com/glfw/glfw/releases/download/3.3/glfw-3.3.zip && unzip glfw-3.3.zip && mkdir -p glfw-3.3/build && cd glfw-3.3/build && cmake .. && make -j4 && make install

WORKDIR /root/repos/class-edge/glfw-3.3
RUN git submodule update --init --recursive && git submodule update --remote --recursive
RUN cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
RUN make -j4