FROM armv7/armhf-ubuntu:16.04

COPY docker/qemu-arm-static /usr/bin/qemu-arm-static

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV VENV /root/venv

RUN apt-get -y update && \
    apt-get -y install git wget htop \
                       nano python-dev python3-dev python-pip \
                       pkg-config apt-utils

# OpenCV + PyGame dependencies
RUN apt-get -y install cmake build-essential arduino-mk zlib1g-dev \
              libsm6 libxext6 libfontconfig1 libxrender1 libglib2.0-0 \
              libpng12-dev libfreetype6-dev \
              libtiff5-dev libjasper-dev libpng12-dev \
              libjpeg-dev libavcodec-dev libavformat-dev \
              libswscale-dev libv4l-dev libgtk2.0-dev \
              libatlas-base-dev gfortran \
              libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev \
              libsdl1.2-dev libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev \
              libzmq3-dev libopenblas-dev libeigen3-dev libffi-dev

RUN pip install virtualenv && \
    virtualenv $VENV --python=python3 && \
    . $VENV/bin/activate && \
    pip install --upgrade pip && \
    pip install enum34==1.1.6 && \
    pip install numpy==1.14.4 && \
    pip install pytest==3.4.2 && \
    pip install pytest-cov==2.5.1 && \
    pip install pyserial==3.4 && \
    pip install pyzmq==16.0.2 && \
    pip install robust-serial==0.1 && \
    pip install six==1.11.0 && \
    pip install tqdm==4.19.5 && \
    pip install ipython && \
    pip install matplotlib && \
    pip install pyyaml setuptools cffi typing && \
    pip install scipy==0.19.1 && \
    pip install scikit-learn==0.19.0 && \
    pip install pygame==1.9.3

ENV PATH=$VENV/bin:$PATH

RUN apt-get -y install unzip

# Compile OpenCV from source
RUN wget https://github.com/opencv/opencv/archive/3.4.3.zip && \
    unzip 3.4.3.zip && \
    rm 3.4.3.zip

ENV OPENCV_DIR opencv-3.4.3

RUN . $VENV/bin/activate && \
    cd $OPENCV_DIR && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D BUILD_EXAMPLES=ON \
    -D WITH_CUDA=OFF \
    -D BUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -D BUILD_opencv_java=OFF \
    -D WITH_EIGEN=ON ..

RUN . $VENV/bin/activate && \
    cd $OPENCV_DIR/build/ && \
    make -j8 && \
    make install && \
    ldconfig

RUN cp /usr/local/lib/python3.5/site-packages/cv2.cpython-35m-arm-linux-gnueabihf.so \
    $VENV/lib/python3.5/site-packages/cv2.so

# Compile PyTorch From Source
RUN git clone --recursive https://github.com/pytorch/pytorch

RUN . $VENV/bin/activate && \
    cd pytorch && \
    MAX_JOBS=8 NO_DISTRIBUTED=1 NO_CAFFE2_OPS=1 NO_CUDA=1 python setup.py install

# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf pytorch && \
    rm -rf $OPENCV_DIR

COPY docker/entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
