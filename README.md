# RacingRobot
Autonomous toy racing car. CAMaleon team at the Toulouse Robot Race 2017.

![The racing robot](https://cdn-images-1.medium.com/max/2000/1*UsmiJ4IzXi6U9svKjB22zw.jpeg)

## Detailed presentation

We wrote an article on medium that detailed our approach. You can read it [here](https://medium.com/@araffin/autonomous-racing-robot-with-an-arduino-a-raspberry-pi-and-a-pi-camera-3e72819e1e63)

## 3D models and training data

### 3D models

3D models (we used onshape.com):
- [Battery Holder](https://cad.onshape.com/documents/a94876919eca38f49abb5ed6/w/b7e86a4005e4b951f0e3d31a/e/5d33c78dde85b96e033f5c9b)
- [Camera Holder](https://cad.onshape.com/documents/dfd040c2f0a4f5410fdcb118/w/d7cc8bc352d92670e7416569/e/0f263a0dde0b04a2688fd3c9)
- [Additional Camera Piece](https://cad.onshape.com/documents/1c4a51d839f2a5989e78ef1f/w/1af5b4b508310461911ecd97/e/a35856fc588eb371f0bac58b)
- [Raspberry Pi Holder](https://cad.onshape.com/documents/621b6943711d60790ddc2b9f/w/c29ba5f453ce625afc8128f6/e/1aa39940e0bdabd3303d76c4)

### Training data

The training data can be downloaded [here](https://www.dropbox.com/s/24x9b6kob5c5847/training_data.zip?dl=0)

There are two folders:
- input_images/ (raw images from remote control)
- label_regions/ (labeled regions of the input images)

The name of the labeled images is as follow: *"x_center"-"y_center"\_"id".jpg*

For example:
- `0-28_452-453r0.jpg`
=> center = (0, 28)
=> id = "452-453r0"

- `6-22_691-23sept1506162644_2073r2.jpg`
=> center = (6, 22)
=> id = "691-23sept1506162644_2073r2"

## How to run everything ?

For installation, see section *Installation*.

## Autonomous mode

Main script, it will try to follow a black&white line.
```
python main.py
```

## Remote Control Mode

0. You need a computer in addition to the raspberry pi
1. Create a Local Wifi Network (e.g. using [create ap](https://github.com/oblique/create_ap))
2. Connect the raspberrypi to this network ([Wifi on RPI](https://www.raspberrypi.org/documentation/configuration/wireless/wireless-cli.md))
3. Launch teleoperation server (it will use the port 5556)
```
python command/python/teleop_server.py
```
4. Launch teleoperation client on your computer
```
python command/python/teleop_client.py
```

## How to train the line detector ?

1. Record a video in the teleoperation mode:
```
python command/python/teleop_server.py -v my_video
```
2. Convert the recorded video from h264 to mp4 using ffmpeg or [MP4Box](https://gpac.wp.imt.fr/mp4box/)
3. Split the video into a sequence of images (please change the paths in the script)
```
python opencv/split_video.py -i video.mp4
```
You have to press enter to pass to the next frame

4. Label the data using the labeling tool (again please change the paths in the script)
```
cd opencv/ (enter opencv folder)
python -m train.label_images
```
To label an image, you have to click on the center of line in the displayed image.
If the image do not contain any line, or if you want to pass to the next frame, press any key.

5. (optional) Augment the dataset using the augmentDataset() function in opencv/train/train.py
6. Train the neural network (again please change the paths in the script)
```
cd opencv/train/ (enter train folder)
python train.py
```
The best model (lowest error on the validation data) will be saved as *mlp_model.npz*.

### Installation
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

OpenCV
- [Guide](http://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/)
- [PreCompiled](https://github.com/jabelone/OpenCV-for-Pi)

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

PySerial
```
sudo apt-get install python-serial
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

ZeroMQ (Message Passing with sockets)
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
pip install --upgrade https://github.com/Theano/Theano/archive/rel-0.10.0beta2.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip install sklearn
```

### Contributors
- Sergio Nicolas Rodriguez Rodriguez
- Antonin Raffin
