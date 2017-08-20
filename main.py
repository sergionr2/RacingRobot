from __future__ import division, print_function

import time
import threading
try:
    import queue

except ImportError:
    import Queue as queue

emptyException = queue.Empty


import numpy as np

import command.python.common as common
from command.python.common import *
from picam.image_analyser import *

THETA_MIN = 60
THETA_MAX = 150
ERROR_MAX = 1.0 # TODO: calibrate max error
MAX_SPEED_STRAIGHT_LINE = 100
MAX_SPEED_SHARP_TURN = 91
MIN_SPEED = 90
# PID Control
Kp = 80
Kd = 0
Ki = 0.0
MAX_ERROR_SECONDS_BEFORE_STOP = 3


def forceStop():
    common.command_queue.put((Order.MOTOR, int(0)))
    # SEND STOP ORDER at the end
    common.resetCommandQueue()
    n_received_semaphore.release()
    n_received_semaphore.release()
    common.command_queue.put((Order.MOTOR, 0))
    common.command_queue.put((Order.SERVO, int((THETA_MIN + THETA_MAX)/2)))

def main_control(out_queue, resolution, n_seconds=5, regions=None):
    """
    :param out_queue: (Queue)
    :param resolution: (int, int)
    :param n_seconds: (int) number of seconds to keep this script alive
    :param regions: [[int]] Regions Of Interest
    """
    start_time = time.time()
    u_angle = 0.
    error, errorD, errorI = 0, 0, 0
    last_error = 0
    turn_percent = 0
    initialized = False
    # Neutral Angle
    theta_init = (THETA_MAX + THETA_MIN) / 2
    angle_order = theta_init
    errors = [False]
    stop_timer = 0
    i = 1
    while time.time() - start_time < n_seconds:
        old_turn_percent = turn_percent
        # Output of image processing
        pts, turn_percent, centroids, errors = out_queue.get()

        # print(centroids)
        # print(errors)

        # Use previous control if we see no line
        if all(errors):
            stop_timer = stop_timer if stop_timer != 0 else time.time()
            if time.time() - stop_timer > MAX_ERROR_SECONDS_BEFORE_STOP:
                forceStop()
            time.sleep(common.rate)
            continue
        stop_timer = 0

        # Compute the error to the center of the line
        error = (resolution[0]//2 - centroids[1,0]) / (resolution[0]//2)


        # Retrieve a and b that define the line
        # a, b = pts
        has_error = any(errors)
        # Reduce max speed if it is a sharp turn
        h = np.clip(turn_percent / 100.0, 0, 1)
        v_max = h * MAX_SPEED_SHARP_TURN + (1 - h) * MAX_SPEED_STRAIGHT_LINE
        # Reduce speed if we have a high error
        t = np.clip(error / float(ERROR_MAX), 0, 1)
        speed_order = t * MIN_SPEED + (1 - t) * v_max

        if initialized:
            errorD = error - last_error
        else:
            initialized = True
        # Update derivative error
        last_error = error

        # PID Control
        u_angle = Kp * error + Kd * errorD + Ki * errorI
        # Update integral error
        errorI += error
        print("error={}".format(error))
        # print("u_angle={}".format(u_angle))

        angle_order = theta_init - u_angle

        angle_order = np.clip(angle_order, THETA_MIN, THETA_MAX).astype(int)
        try: 
            if i % 2 == 0:
                i = 1
                common.command_queue.put_nowait((Order.SERVO, angle_order))
            else:
                i = 2
                common.command_queue.put_nowait((Order.MOTOR, int(speed_order)))
        except Exception as e:
            print(e)
            pass
        # print("angle order = {}".format(angle_order))

    # SEND STOP ORDER at the end
    forceStop()
    # Make sure STOP order is sent
    time.sleep(0.2)

if __name__ == '__main__':
    try:
        serial_port = get_serial_ports()[0]
        serial_file = serial.Serial(port=serial_port, baudrate=BAUDRATE, timeout=0, writeTimeout=0)
    except Exception as e:
        raise e

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

    print("Connected to Arduino")
    resolution = (640//2, 480//2)
    max_width = resolution[0]
    # Regions of interest
    r0 = [0, 150, max_width, 50]
    r1 = [0, 125, max_width, 25]
    r2 = [0, 100, max_width, 25]
    regions = [r0, r1, r2]
    # image processing queue, output centroids
    out_queue = queue.Queue()
    condition_lock = threading.Lock()
    exit_condition = threading.Condition(condition_lock)
    image_thread = ImageProcessingThread(Viewer(out_queue, resolution, debug=False, fps=40), exit_condition)

    threads = [CommandThread(serial_file, command_queue),
               ListenerThread(serial_file), image_thread]
    for t in threads:
        t.start()

    time.sleep(1)

    main_control(out_queue, resolution=resolution, n_seconds=20, regions=regions)

    common.exit_signal = True
    n_received_semaphore.release()

    print("EXIT")
    # End the thread
    with exit_condition:
        exit_condition.notify_all()

    for t in threads:
        t.join()
