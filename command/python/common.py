from __future__ import print_function, with_statement, division, unicode_literals

import sys
import glob
import os
import struct
import threading
try:
    import queue
except ImportError:
    import Queue as queue
import time
from enum import Enum

import serial

BAUDRATE = 115200
exit_signal = False
is_connected_lock = threading.Lock()
is_connected = False
n_received_semaphore = threading.Semaphore(1)
serial_lock = threading.Lock()
command_queue = queue.Queue(1)
rate = 1/90 # 90 fps

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
    return struct.unpack('<b', bytearray(f.read(1)))[0]

def readTwoBytesInt(f):
    """
    :param f: file handler or serial file
    :return: (int16_t)
    """
    return struct.unpack('<h', bytearray(f.read(2)))[0]

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

class CommandThread(threading.Thread):
    def __init__(self, serial_file, command_queue):
        threading.Thread.__init__(self)
        self.deamon = True
        self.serial_file = serial_file
        self.command_queue = command_queue

    def run(self):
        while not exit_signal:
            n_received_semaphore.acquire()
            # Wait until connected
            if exit_signal:
                break
            try:
                order, param = self.command_queue.get_nowait()
            except queue.Empty:
                time.sleep(rate)
                n_received_semaphore.release()
                continue

            with serial_lock:
                sendOrder(self.serial_file, order.value)
                # print("Sent {}".format(order))
                if order == Order.MOTOR:
                    writeOneByteInt(self.serial_file, param)
                elif order == Order.SERVO:
                    writeTwoBytesInt(self.serial_file, param)
            time.sleep(rate)

class ListenerThread(threading.Thread):
    def __init__(self, serial_file):
        threading.Thread.__init__(self)
        self.deamon = True
        self.serial_file = serial_file

    def run(self):
        while not exit_signal:
            bytes_array = bytearray(self.serial_file.read(1))
            if not bytes_array:
               time.sleep(rate)
               continue
            byte = bytes_array[0]
            with serial_lock:
                try:
                    order = Order(byte)
                except ValueError:
                    continue
                if order == Order.RECEIVED:
                   n_received_semaphore.release()
                decodeOrder(self.serial_file, byte)
            time.sleep(rate)
        print("Listener thread exited")
