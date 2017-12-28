# Autonomous Racing Robot With an Arduino, a Raspberry Pi and a Pi Camera
Autonomous toy racing car. CAMaleon team at the Toulouse Robot Race 2017. Medium article: [https://medium.com/@araffin/autonomous-racing-robot-with-an-arduino-a-raspberry-pi-and-a-pi-camera-3e72819e1e63](https://medium.com/@araffin/autonomous-racing-robot-with-an-arduino-a-raspberry-pi-and-a-pi-camera-3e72819e1e63)

**Video of the car**: [https://www.youtube.com/watch?v=xhI71ZdSh6k](https://www.youtube.com/watch?v=xhI71ZdSh6k)

[![The racing robot](https://cdn-images-1.medium.com/max/2000/1*UsmiJ4IzXi6U9svKjB22zw.jpeg)](https://www.youtube.com/watch?v=xhI71ZdSh6k)

## Detailed Presentation

We wrote an article on medium that detailed our approach. You can read it [here](https://medium.com/@araffin/autonomous-racing-robot-with-an-arduino-a-raspberry-pi-and-a-pi-camera-3e72819e1e63)

## 3D Models and Training Data

### 3D Models

3D models (we used onshape.com):
- [Battery Holder](https://cad.onshape.com/documents/a94876919eca38f49abb5ed6/w/b7e86a4005e4b951f0e3d31a/e/5d33c78dde85b96e033f5c9b)
- [Camera Holder](https://cad.onshape.com/documents/dfd040c2f0a4f5410fdcb118/w/d7cc8bc352d92670e7416569/e/0f263a0dde0b04a2688fd3c9)
- [Additional Camera Piece](https://cad.onshape.com/documents/1c4a51d839f2a5989e78ef1f/w/1af5b4b508310461911ecd97/e/a35856fc588eb371f0bac58b)
- [Raspberry Pi Holder](https://cad.onshape.com/documents/621b6943711d60790ddc2b9f/w/c29ba5f453ce625afc8128f6/e/1aa39940e0bdabd3303d76c4)

### Training Data

**Outdated** (you have to use convert_old_format.py to use current code, now all the informations are in a pickle file)

The training data (7600+ labeled images) can be downloaded [here](https://www.dropbox.com/s/24x9b6kob5c5847/training_data.zip?dl=0)

There are two folders:
- input_images/ (raw images from remote control)
- label_regions/ (labeled regions of the input images)

The name of the labeled images is as follow: **"x_center"-"y_center"\_"id".jpg**

For example:
- `0-28_452-453r0.jpg`
=> center = (0, 28)
| id = "452-453r0"

- `6-22_691-23sept1506162644_2073r2.jpg`
=> center = (6, 22)
| id = "691-23sept1506162644_2073r2"

## How to run everything ?

For installation, see section **Installation**.

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
python command/python/teleop_server.py
```
4. Launch teleoperation client on your computer (you have to edit the raspberry pi `IP` in the code)
```
python command/python/teleop_client.py
```
5. Enjoy! You can now control the car with the keyboard.

## How to train the line detector ?

1. Record a video in the teleoperation mode:
```
python command/python/teleop_server.py -v my_video
```
2. Convert the recorded video from h264 to mp4 using ffmpeg or [MP4Box](https://gpac.wp.imt.fr/mp4box/)
```
MP4Box -add video.h264 video.mp4
```

3. Split the video into a sequence of images
```
python -m train.split_video -i video.mp4 --no-display -o path/output/folder
```

4. Label the data using the labeling tool
```
python -m train.label_images -i path/to/input/folder -o path/to/output/folder
```
To label an image, you have to click on the center of line in the displayed image.
If the image do not contain any line, or if you want to pass to the next frame, press any key.

5. Train the neural network (again please change the paths in the script)
```
python -m train.train -f path/input/folder
```
The best model (lowest error on the validation data) will be saved as *mlp_model_tmp.npz*.


6. Test the trained neural network

```
python -m train.test -f path/input/folder -w mlp_model_tmp
```

### Installation

#### Recommended : Use an image with everything already installed

0. You need a micro sd card (warning, all data on that card will be overwritten)

1. Download the image [here](https://drive.google.com/open?id=0Bz4VOC2vLbgPTl9LZzNNcnBCWUU)

Infos about the linux image:
OS: [Ubuntu MATE 16.04](https://ubuntu-mate.org/raspberry-pi/) for raspberry pi

**Username**: enstar

**Password**: enstar


Installed softwares:
 - all the dependencies for that project (OpenCV 3.2.0, PyTorch, ...)
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
```
pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl
pip install torchvision

pip install sklearn # or sudo apt-get install python-sklearn
```

### Contributors
- Sergio Nicolas Rodriguez Rodriguez
- Antonin Raffin
