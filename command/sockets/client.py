from __future__ import print_function, with_statement, division

import zmq
import random
import sys
import time

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:%s" % port)

while True:
    msg = socket.recv_string()
    print(msg)
    socket.send_string("client message to server1")
    socket.send_string("client message to server2")
    time.sleep(1)
    socket.send_string("stop")
