from __future__ import print_function, with_statement, division

import zmq
import random
import sys
import time

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:{}".format(port))

while True:
    socket.send_string("Server message to client3")
    msg = socket.recv_string()
    print(msg)
    if msg == "stop":
        socket.close()
        break
    time.sleep(1)
