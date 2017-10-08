from __future__ import division, print_function

import signal
import time
import threading

# Python 2/3 support
try:
    import queue
except ImportError:
    import Queue as queue

import serial
import numpy as np

import command.python.common as common
from command.python.common import is_connected, n_received_semaphore, command_queue,\
                                  CommandThread, ListenerThread, sendOrder, Order, get_serial_ports, BAUDRATE
from picam.image_analyser import ImageProcessingThread, Viewer

emptyException = queue.Empty
fullException = queue.Full

THETA_MIN = 70  # value in [0, 255] sent to the servo
THETA_MAX = 150
ERROR_MAX = 1.0
MAX_SPEED_STRAIGHT_LINE = 50  # order between 0 and 100
MAX_SPEED_SHARP_TURN = 15
MIN_SPEED = 10
# PID Control
Kp_turn = 40
Kp_line = 35
Kd = 30
Ki = 0.0
FPS = 60
N_SECONDS = 77  # number of seconds before exiting the program
alpha = 0.8  # alpha of the moving mean for the turn coefficient


def forceStop():
    # SEND STOP ORDER at the end
    common.resetCommandQueue()
    n_received_semaphore.release()
    n_received_semaphore.release()
    common.command_queue.put((Order.MOTOR, 0))
    common.command_queue.put((Order.SERVO, int((THETA_MIN + THETA_MAX) / 2)))


def main_control(out_queue, resolution, n_seconds=5):
    """
    :param out_queue: (Queue)
    :param resolution: (int, int)
    :param n_seconds: (int) number of seconds to keep this script alive
    """
    mean_h = 0
    start_time = time.time()
    error, errorD, errorI = 0, 0, 0
    last_error = 0
    initialized = False
    # Neutral Angle
    theta_init = (THETA_MAX + THETA_MIN) / 2
    # Use mutable to be modified by signal handler
    should_exit = [False]

    # Stop the robot on ctrl+c and exit the script
    def ctrl_c(signum, frame):
        print("STOP")
        should_exit[0] = True

    signal.signal(signal.SIGINT, ctrl_c)
    last_time = time.time()

    while time.time() - start_time < n_seconds and not should_exit[0]:
        # Output of image processing
        turn_percent, centroids = out_queue.get()
        # print(centroids)
        # Compute the error to the center of the line
        # Here we use the farthest centroids
        error = (resolution[0] // 2 - centroids[-1, 0]) / (resolution[0] // 2)

        # Reduce max speed if it is a sharp turn
        h = np.clip(turn_percent / 100.0, 0, 1)
        # Moving mean
        mean_h += alpha * (h - mean_h)

        # print("mean_h={}".format(mean_h))
        h = mean_h
        v_max = h * MAX_SPEED_SHARP_TURN + (1 - h) * MAX_SPEED_STRAIGHT_LINE

        Kp = h * Kp_turn + (1 - h) * Kp_line

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
        # TODO: add dt in the equation
        dt = time.time() - last_time
        u_angle = Kp * error + Kd * errorD + Ki * errorI
        # Update integral error
        errorI += error
        last_time = time.time()
        # print("error={}".format(error))
        # print("u_angle={}".format(u_angle))

        angle_order = theta_init - u_angle
        angle_order = np.clip(angle_order, THETA_MIN, THETA_MAX).astype(int)

        try:
            common.command_queue.put_nowait((Order.MOTOR, int(speed_order)))
            common.command_queue.put_nowait((Order.SERVO, angle_order))
        except fullException:
            print("Queue is full")

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

    print("Connected to Arduino")
    resolution = (640 // 2, 480 // 2)
    max_width = resolution[0]

    # image processing queue, output centroids
    out_queue = queue.Queue()
    condition_lock = threading.Lock()
    exit_condition = threading.Condition(condition_lock)

    print("Starting Image Processing Thread")
    image_thread = ImageProcessingThread(Viewer(out_queue, resolution, debug=False, fps=FPS), exit_condition)
    # Wait for camera warmup
    time.sleep(1)

    print("Starting Communication Threads")
    # Threads for arduino communication
    threads = [CommandThread(serial_file, command_queue),
               ListenerThread(serial_file), image_thread]
    for t in threads:
        t.start()

    print("Starting Control Thread")
    main_control(out_queue, resolution=resolution, n_seconds=N_SECONDS)

    common.exit_signal = True
    n_received_semaphore.release()

    print("Exiting...")
    # End the thread
    with exit_condition:
        exit_condition.notify_all()

    for t in threads:
        t.join()
