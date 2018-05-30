# Autonomous Racing Robot With an Arduino, a Raspberry Pi and a Pi Camera

[![Build Status](https://travis-ci.org/araffin/RacingRobot.svg?branch=master)](https://travis-ci.org/araffin/RacingRobot)

Autonomous toy racing car. CAMaleon team at the Toulouse Robot Race 2017. Humbavision team at IronCar.
Medium article: [https://medium.com/@araffin/autonomous-racing-robot-with-an-arduino-a-raspberry-pi-and-a-pi-camera-3e72819e1e63](https://medium.com/@araffin/autonomous-racing-robot-with-an-arduino-a-raspberry-pi-and-a-pi-camera-3e72819e1e63)

**Video of the car**: [https://www.youtube.com/watch?v=xhI71ZdSh6k](https://www.youtube.com/watch?v=xhI71ZdSh6k)

[![The racing robot](https://cdn-images-1.medium.com/max/2000/1*UsmiJ4IzXi6U9svKjB22zw.jpeg)](https://www.youtube.com/watch?v=xhI71ZdSh6k)

Table of Contents
=================
  * [Detailed Presentation](#detailed-presentation)
  * [3D Models and Training Data](#3d-models-and-training-data)
    * [3D Models](#3d-models)
    * [Training Data](#training-data)
      * [IronCar and Toulouse Robot Race Datasets](#ironcar-and-toulouse-robot-race-datasets)
  * [How to run everything ?](#how-to-run-everything-)
  * [Autonomous mode](#autonomous-mode)
  * [Remote Control Mode](#remote-control-mode)
  * [How to train the line detector ?](#how-to-train-the-line-detector-)
    * [Benchmark](#benchmark)
  * [Installation](#installation)
    * [Recommended : Use an image with everything already installed](#recommended--use-an-image-with-everything-already-installed)
    * [From Scratch](#from-scratch)
    * [Python Packages](#python-packages)
  * [C++ Extension](#c-extension)
  * [Contributors](#contributors)

## Detailed Presentation

We wrote an article on medium that detailed our approach. You can read it [here](https://medium.com/@araffin/autonomous-racing-robot-with-an-arduino-a-raspberry-pi-and-a-pi-camera-3e72819e1e63)

En français: [http://enstar.ensta-paristech.fr/blog/public/racing_car/](http://enstar.ensta-paristech.fr/blog/public/racing_car/)

## 3D Models and Training Data

### 3D Models

3D models (we used onshape.com):
- [Battery Holder](https://cad.onshape.com/documents/a94876919eca38f49abb5ed6/w/b7e86a4005e4b951f0e3d31a/e/5d33c78dde85b96e033f5c9b)
- [Camera Holder](https://cad.onshape.com/documents/dfd040c2f0a4f5410fdcb118/w/d7cc8bc352d92670e7416569/e/0f263a0dde0b04a2688fd3c9)
- [Additional Camera Piece](https://cad.onshape.com/documents/1c4a51d839f2a5989e78ef1f/w/1af5b4b508310461911ecd97/e/a35856fc588eb371f0bac58b)
- [Raspberry Pi Holder](https://cad.onshape.com/documents/621b6943711d60790ddc2b9f/w/c29ba5f453ce625afc8128f6/e/1aa39940e0bdabd3303d76c4)

Note: the Battery Holder was designed for this [External Battery](https://www.amazon.fr/gp/product/B00Y7S4JRQ/ref=oh_aui_detailpage_o00_s01?ie=UTF8&psc=1)


### Training Data

#### IronCar and Toulouse Robot Race Datasets


We release the different videos taken with the on-board camera, along we the labeled data (the labels are in a pickle file) for IronCar and Toulouse Robot Race:

- [Videos](https://drive.google.com/open?id=1VJ46uBZUfxwUVPHGw1p8d6cgHDGervcL)
- (outdated) [Toulouse Dataset](https://drive.google.com/open?id=1vj7N0aE-eyKg7OY0ZdlkwW2rl51UzwEW)
- (outdated) [IronCar Dataset](https://drive.google.com/open?id=1FZdXnrO7WAo4A4-repE_dglCc2ZoAJwa)


## How to run everything ?

For installation, see section **[Installation](#installation)**.

## Autonomous mode

0. Compile and upload the code on the Arduino
```
cd arduino/
make
make upload
```

1. Launch the main script on the Raspberry Pi, it will try to follow a black&white line.
All useful constants can be found in `constants.py`.
```
python main.py
```

## Remote Control Mode

0. You need a computer in addition to the raspberry pi
1. Create a Local Wifi Network (e.g. using [create ap](https://github.com/oblique/create_ap))
2. Connect the raspberry pi to this network ([Wifi on RPI](https://www.raspberrypi.org/documentation/configuration/wireless/wireless-cli.md))
3. Launch teleoperation server (it will use the port 5556)
```
python teleop/teleop_server.py
```
4. Launch teleoperation client on your computer (you have to edit the raspberry pi `IP` in the code)
```
python teleop/teleop_client.py
```
5. Enjoy! You can now control the car with the keyboard.

## How to train the line detector ?

1. Record a video in the teleoperation mode:
```
python teleop/teleop_server.py -v my_video
```
2. Convert the recorded video from h264 to mp4 using ffmpeg or [MP4Box](https://gpac.wp.imt.fr/mp4box/)
```
MP4Box -add video.h264 video.mp4
```

3. Split the video into a sequence of images
```
python -m train.split_video -i video.mp4 --no-display -o path/to/dataset/folder
```

4. Label the data using the labeling tool: [https://github.com/araffin/graph-annotation-tool](https://github.com/araffin/graph-annotation-tool)


5. Rename the json file that contains the labels to `labels.json` and put it in the same folder of the dataset (folder with the images)

5. Train the neural network (again please change the paths in the script)
```
python -m train.train -f path/to/dataset/folder
```
The best model (lowest error on the validation data) will be saved as *cnn_model_tmp.pth*.


6. Test the trained neural network

```
python -m train.test -f path/to/dataset/folder -w cnn_model_tmp.pth
```

### Benchmark

For profiling 5000 iterations of image processing:
```
python -m opencv.benchmark -i path/to/input/image.jpg -n 5000
```

## Installation

#### Recommended : Use an image with everything already installed

0. You need a 16GB micro sd card (warning, all data on that card will be overwritten)
WARNING: for a smaller sd card, you need to resize the image before writing it (this [link](https://github.com/billw2/rpi-clone) may help)

1. Download the image [here](https://drive.google.com/open?id=1CUmSKOQ7i_XTrsLCRntypK9KcVaVwM4h)

Infos about the linux image:
OS: [Ubuntu MATE 16.04](https://ubuntu-mate.org/raspberry-pi/) for raspberry pi

**Username**: enstar

**Password**: enstar


Installed softwares:
 - all the dependencies for that project (OpenCV >= 3.1, PyTorch, ...)
 - the current project (in the folder RacingRobot/)
 - ROS Kinetic

Camera and ssh are enabled.


2. Identify the name of your sd card using:
```
fdisk -l
```
For instance, it gives:
```
/dev/mmcblk0p1            2048   131071   129024   63M  c W95 FAT32 (LBA)
/dev/mmcblk0p2          131072 30449663 30318592 14,5G 83 Linux
```
In that case, your sd card is named */dev/mmcblk0* (p1 and p2 stand for partition).

3. Write the downloaded image on the sd card.
```
gunzip --stdout ubuntu_ros_racing_robot.img.gz | sudo dd bs=4M of=/dev/mmcblk0
```

4. Enjoy!
The current project is located in `RacingRobot/`. There is also a ROS version of the remote control in `catkin_ws/src/`.


If you want to back up an image of a raspberry pi:
```
sudo dd bs=4M if=/dev/mmcblk0 | gzip > ubuntu_ros_racing_robot.img.gz
```

#### From Scratch

Update your pi
```
sudo apt-get update
sudo apt-get upgrade
sudo rpi-update
```

Arduino + Arduino Makefile + rlwrap + screen
```
sudo apt-get install arduino-core arduino-mk rlwrap screen
```

- Arduino 1.0.5
- [Arduino-Makefile](https://github.com/sudar/Arduino-Makefile)
- OpenCV 3.1
- libserial-dev (apt-get)
- Python 2 or 3

OpenCV
- [PreCompiled](https://github.com/jabelone/OpenCV-for-Pi) This is the **recommended method**
- [Guide](http://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/)

Libserial (apt-get or compile from source)
- [LibSerial](https://github.com/crayzeewulf/libserial)
- [Boost](http://www.boost.org/)
- [SIP](http://pyqt.sourceforge.net/Docs/sip4/installation.html)

```
# Boost
sudo apt-get install libboost-all-dev

# After libserial installation:
sudo ldconfig
```

#### Python Packages
All the required packages can be found in `requirements.txt`

PySerial
```
sudo pip install pyserial
# Add user to dialout group to have the right to write on the serial port
sudo usermod -a -G dialout $USER
# You need to logout/login again for that change to be taken into account
```

TQDM (for progressbar)
```
pip install tqdm
```

[PyGame](http://www.pygame.org/wiki/CompileUbuntu#Installing%20pygame%20with%20pip)
For teleoperation
```
pip install pygame
```

Enum for python 2
```
pip install enum34
```

[Wifi on RPI](https://www.raspberrypi.org/documentation/configuration/wireless/wireless-cli.md)

ZeroMQ (Message Passing with sockets) for remote control mode
```
sudo apt-get install libzmq3-dev
pip install pyzmq
```
or
```
git clone https://github.com/zeromq/libzmq/
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
pip install pyzmq
```

Additional python dev-dependencies for training the neural network:
On your laptop:
```
pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl
pip install torchvision

pip install sklearn # or sudo apt-get install python-sklearn
```

On the raspberry pi :

- You can try to use Python 2 wheel (not tested) that was created for this project:

0. Download Python Wheel [here](https://drive.google.com/open?id=1vFV73nZDbZ1GDRzz4YeBGDbc4S5Ddxwr)

And then:
```
pip install torch-0.4.0a0+b23fa21-cp27-cp27mu-linux_armv7l.whl
```

Or follow this tutorial:
[PyTorch on the raspberry pi](http://book.duckietown.org/fall2017/duckiebook/pytorch_install.html)

0. Make sure you have at least 3 Go of Swap. (see link above)

1. (optional) Install a recent version of cmake + scikit-build + ninja

2. Install PyTorch

```
# don't forget to set the env variables:
export NO_CUDA=1
export NO_DISTRIBUTED=1
git clone --recursive https://github.com/pytorch/pytorch
sudo -EH python setup.py install
# torchvision is not used yet
sudo -H pip install torchvision
```

### C++ Extension

Please read [opencv/c_extension/README.md](https://github.com/sergionr2/RacingRobot/tree/master/opencv/c_extension) for more information.

### Contributors
- Sergio Nicolas Rodriguez Rodriguez
- Antonin Raffin
