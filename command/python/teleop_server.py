from __future__ import print_function, with_statement, division

import zmq
import random
import sys
import time

import common
from common import *

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:{}".format(port))


try:
    serial_port = get_serial_ports()[0]
    serial_file = serial.Serial(port=serial_port, baudrate=BAUDRATE, timeout=0, writeTimeout=0)
except Exception as e:
    raise e
# serial_file_write = open("test.log", 'ab')
# serial_file_read = open("test.log", 'rb')

while not is_connected:
    print("Waiting for arduino...")
    sendOrder(serial_file, Order.HELLO.value)
    bytes_array = bytearray(serial_file.read(1))
    if not bytes_array:
        time.sleep(2)
        continue
    byte = bytes_array[0]
    if byte in [Order.HELLO.value, Order.ALREADY_CONNECTED.value]:
        is_connected = True
    time.sleep(1)

threads = [CommandThread(serial_file, command_queue),
           ListenerThread(serial_file)]
for t in threads:
    t.start()

print("Connected to Arduino, waiting for client...")
socket.send(b'1')
print("Connected To Client")
i = 0
while True:
    control_speed, angle_order = socket.recv_json()
    #control_speed, angle_order = int(control_speed), int(angle_order)
    print(control_speed, angle_order)
    try:
        if i%2 == 0:
            i = 1
            common.command_queue.put_nowait((Order.MOTOR, control_speed))
            common.command_queue.put_nowait((Order.SERVO, angle_order))
        else:
            i = 2
            common.command_queue.put_nowait((Order.SERVO, angle_order))
            common.command_queue.put_nowait((Order.MOTOR, control_speed))
    except Exception as e:
    	pass

    if control_speed == -999:
        socket.close()
        break
# TODO: SEND STOP ORDER at the end
common.exit_signal = True
n_received_semaphore.release()
print("EXIT")

for t in threads:
    t.join()