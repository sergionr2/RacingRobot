import sys
import glob
import os
import struct
from enum import Enum

import serial

BAUDRATE = 115200

class Order(Enum):
    HELLO = 0
    SERVO = 1
    MOTOR = 2
    ALREADY_CONNECTED = 3
    ERROR = 4
    RECEIVED = 5
    STOP = 6

def get_serial_ports():
    """
    Lists serial ports
    :return: [str] A list of available serial ports
    """
    ports = glob.glob('/dev/tty[A-Za-z]*')
    results = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            results.append(port)
        except (OSError, serial.SerialException):
            pass
    return results

def readOneByteInt(f):
    """
    :param f: file handler or serial file
    :return: (int8_t)
    """
    return struct.unpack('<b', f.read(1))[0]

def readTwoBytesInt(f):
    """
    :param f: file handler or serial file
    :return: (int16_t)
    """
    return struct.unpack('<h', f.read(2))[0]

def writeOneByteInt(f, value):
    """
    :param f: file handler or serial file
    :param value: (int8_t)
    """
    f.write(struct.pack('<b', value))
    f.flush()

# Alias
sendOrder = writeOneByteInt

def writeTwoBytesInt(f, value):
    """
    :param f: file handler or serial file
    :param value: (int16_t)
    """
    f.write(struct.pack('<h', value))
    f.flush()
