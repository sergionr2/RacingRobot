# RacingRobot
Autonomous toy racing car

## Requirements

- Arduino 1.0.5
- [Arduino-Makefile](https://github.com/sudar/Arduino-Makefile)
- OpenCV
- libserial-dev (apt-get)

### Installation
Update your pi
```
sudo apt-get update
sudo apt-get upgrade
sudo rpi-update
```

Arduino + Arduino Makefile + rlwrap
```
sudo apt-get install arduino-core arduino-mk rlwrap
```

OpenCV
- [Guide](http://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/)
- [PreCompiled](https://github.com/jabelone/OpenCV-for-Pi)

Python binding
```
sudo apt-get install python-opencv
```

Libserial
- [LibSerial](https://github.com/crayzeewulf/libserial)
- [Boost](http://www.boost.org/)
- [SIP](http://pyqt.sourceforge.net/Docs/sip4/installation.html)
```
# Boost
sudo apt-get install libboost-all-dev
```

Raspicam
- [Userland](https://github.com/raspberrypi/userland)
- [Raspicam](https://www.uco.es/investiga/grupos/ava/node/40)


[Wifi on RPI](https://www.raspberrypi.org/documentation/configuration/wireless/wireless-cli.md)

### Contributors
- Sergio Nicolas Rodriguez Rodriguez
- Antonin Raffin
