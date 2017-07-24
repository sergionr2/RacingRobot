import sys
import glob
import os
import struct
from enum import Enum

import serial

class Order(Enum):
    HELLO = 0
    SERVO = 1
    MOTOR = 2
    ALREADY_CONNECTED = 3
    ERROR = 4
    RECEIVED = 5
    STOP = 6

def serial_ports():
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

# Automatically find the right port (ex: '/dev/ttyACM0')
# serial_file = serial_ports()[0]
# serial_port = serial.Serial(port=serial_file, baudrate=115200, timeout=0, writeTimeout=0)


def decodeOrder(f, byte):
    """
    :param f: file handler or serial file
    :param byte: (int8_t)
    """
    order = Order(byte)
    if order == Order.HELLO:
        print("hello")
    elif order == Order.SERVO:
        angle = readTwoBytesInt(f)
        # Bit representation
        # print('{0:016b}'.format(angle))
        print("servo {}".format(angle))
    elif order == Order.MOTOR:
        speed = readOneByteInt(f)
        print("motor {}".format(speed))
    elif order == Order.ALREADY_CONNECTED:
        print("ALREADY_CONNECTED")
    elif order == Order.ERROR:
        code_error = readTwoBytesInt(f)
        print("Error {}".format(code_error))
    elif order == Order.RECEIVED:
        print("RECEIVED")
    elif order == Order.STOP:
        print("STOP")
    else:
        print("unknown order", byte)

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

def writeTwoBytesInt(f, value):
    """
    :param f: file handler or serial file
    :param value: (int16_t)
    """
    f.write(struct.pack('<h', value))

filename = "test2.log"
with open(filename, 'rb') as f:
    while True:
       byte_s = f.read(1)
       if not byte_s:
          break
       byte = byte_s[0]
       decodeOrder(f, byte)

with open("test2.log", "wb") as f:
    for order in Order:
        writeOneByteInt(f, order.value)
        if order == Order.HELLO:
            pass
        elif order == Order.SERVO:
            angle = 3500
            writeTwoBytesInt(f, angle)
        elif order == Order.MOTOR:
            speed = 100
            writeOneByteInt(f, speed)
        elif order == Order.ALREADY_CONNECTED:
            pass
        elif order == Order.ERROR:
            code_error = 404
            writeTwoBytesInt(f, code_error)
        elif order == Order.RECEIVED:
            pass
        elif order == Order.STOP:
            pass
        else:
            pass
