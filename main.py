from __future__ import division, print_function

import time
import threading
try:
    import queue
except ImportError:
    import Queue as queue

import numpy as np

import command.python.common as common
from command.python.common import *
from image_analyser import *

THETA_MIN = 60
THETA_MAX = 130
ERROR_MAX = 0.8 # TODO: calibrate max error
MAX_SPEED_STRAIGHT_LINE = 100
MAX_SPEED_SHARP_TURN = 60
MIN_SPEED = 30
# PID Control
Kp = 0.2
Kd = 0.1
Ki = 0.0
MAX_ERROR_SECONDS_BEFORE_STOP = 3

def forceStop():
    with common.command_queue.mutex:
        common.command_queue.clear()
        # Release the command queue
        n_received_semaphore.release()
        common.command_queue.put((Order.STOP, 0))

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
    angle_order = (THETA_MAX + THETA_MIN) / 2
    errors = [False]
    stop_timer = 0

    while time.time() - start_time < n_seconds:
        old_turn_percent = turn_percent
        # Output of image processing
        pts, turn_percent, centroids, errors = out_queue.get()

        # Use previous control if we see no line
        if all(errors):
            stop_timer = stop_timer if stop_timer != 0 else time.time()
            if time.time() - stop_timer > MAX_ERROR_SECONDS_BEFORE_STOP:
                forceStop()
            continue
        stop_timer = 0

        # Compute the error to the center of the line
        error = (resolution[0]//2 - centroids[-1,0]) / (resolution[0]//2)
        # Retrieve a and b that define the line
        # a, b = pts
        has_error = any(errors)
        # Reduce max speed if it is a sharp turn
        h = constrain(turn_percent / 100.0, 0, 1)
        v_max = h * MAX_SPEED_SHARP_TURN + (1 - h) * MAX_SPEED_STRAIGHT_LINE
        # Reduce speed if we have a high error
        t = constrain(error / float(ERROR_MAX), 0, 1)
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

        angle_order += u_angle
        angle_order = np.clip(angle_order, THETA_MIN, THETA_MAX).astype(int)
        common.command_queue.put((Order.SERVO, angle_order))
        # common.command_queue.put((Order.MOTOR, int(speed_order)))

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

    resolution = (640//2, 480//2)
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

    main_control(out_queue, resolution=resolution, n_seconds=1*30, regions=regions)

    common.exit_signal = True
    n_received_semaphore.release()

    print("EXIT")
    # End the thread
    with exit_condition:
        exit_condition.notify_all()

    for t in threads:
        t.join()
