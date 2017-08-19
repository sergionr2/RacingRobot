from __future__ import print_function, with_statement, division

import zmq
import random
import sys
import time

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:{}".format(port))
socket.send(b'1')

while True:
    msg = socket.recv_json()
    if msg[0] == -999:
        socket.close()
        break
    print(msg)
