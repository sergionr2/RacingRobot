from __future__ import print_function, with_statement, division

import zmq
import random
import sys
import time
import common

from teleop import *

# stdscr: main window
def main(stdscr, socket):
    interface = Interface(stdscr)
    x, theta, status, count = 0, 0, 0, 0
    control_speed, control_turn = 0, 0
    start_time = time.time()
    counter = 0
    while True:
        keycode = interface.readKey()
        counter += 1
        info = "{:.2f} fps".format(counter/(time.time()-start_time))

        publish(interface, control_speed, control_turn, info)
        if keycode in moveBindings.keys():
            x, theta = moveBindings[keycode]
            count = 0
        elif keycode == ord('k') or keycode == KEY_CODE_SPACE:
                x, theta = 0, 0
                control_speed, control_turn = 0, 0
        elif keycode == ord('q'):
            break
        else:
            count += 1
            if count > 4:
                x, theta = 0, 0

        # Smooth control
        control_speed, control_turn = control(x, theta, control_speed, control_turn)
        sendToServer(socket, control_speed, control_turn)
        # force 40 fps
        time.sleep(1/40.0)

def sendToServer(socket, control_speed, control_turn):
    """
    :param control_speed: (float)
    :param control_turn: (float)
    """
    # Send Orders
    t = (control_turn + MAX_TURN) / (2 * MAX_TURN)
    angle_order = int(THETA_MIN  * t + THETA_MAX * (1 - t))
    socket.send_json((control_speed, angle_order))


port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:{}".format(port))

msg = socket.recv()
print("Connected To Server")
try:
    curses.wrapper(main, socket)
except KeyboardInterrupt as e:
    pass
finally:
    socket.send_json((-999, -999))
    socket.close()
