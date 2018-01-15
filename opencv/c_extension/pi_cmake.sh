#!/bin/bash
cd build \
 && cmake -DPYTHON_EXECUTABLE=/usr/bin/python2.7 \
 		  -DPYTHON_LIBRARY=/usr/lib/arm-linux-gnueabihf/libpython2.7.so\
 		  -DPYTHON_INCLUDE_DIR=/usr/include/python2.7 ..\
 && make install