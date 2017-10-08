# RacingRobot
Autonomous toy racing car

## Requirements

- Arduino 1.0.5
- [Arduino-Makefile](https://github.com/sudar/Arduino-Makefile)
- OpenCV 3.1
- libserial-dev (apt-get)

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

OpenCV
- [Guide](http://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/)
- [PreCompiled](https://github.com/jabelone/OpenCV-for-Pi)

Libserial
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
