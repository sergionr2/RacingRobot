"""
Main script for autonomous mode
It launches all the thread and does the PID control
"""
from __future__ import division, print_function

import signal
import threading
import time

# Python 2/3 support
try:
    import queue
except ImportError:
    import Queue as queue

import serial
import numpy as np
from tqdm import tqdm

from command.python.common import is_connected, writeOneByteInt, writeTwoBytesInt, \
    sendOrder, Order, get_serial_ports, BAUDRATE
from picam.image_analyser import ImageProcessingThread, Viewer
from constants import THETA_MIN, THETA_MAX, ERROR_MAX, MAX_SPEED_SHARP_TURN, MAX_SPEED_STRAIGHT_LINE, \
    MIN_SPEED, Kp_turn, Kp_line, Kd, Ki, FPS, N_SECONDS, ALPHA, CAMERA_RESOLUTION

emptyException = queue.Empty
fullException = queue.Full


def sendOrders(speed_order, angle_order):
    """
    Send orders to the arduino
    :param speed_order: (float)
    :param angle_order: (int)
    """
    sendOrder(serial_file, Order.MOTOR.value)
    writeOneByteInt(serial_file, int(speed_order))
    sendOrder(serial_file, Order.SERVO.value)
    writeTwoBytesInt(serial_file, angle_order)


def forceStop():
    # SEND STOP ORDER at the end
    sendOrders(speed_order=0, angle_order=int((THETA_MIN + THETA_MAX) / 2))


def main_control(out_queue, resolution, n_seconds=5):
    """
    :param out_queue: (Queue)
    :param resolution: (int, int)
    :param n_seconds: (int) number of seconds to keep this script alive
    """
    # Moving mean for line curve estimation
    mean_h = 0
    start_time = time.time()
    error, errorD, errorI = 0, 0, 0
    last_error = 0
    # Neutral Angle
    theta_init = (THETA_MAX + THETA_MIN) / 2
    # Middle of the image
    x_center = resolution[0] // 2
    max_error_px = resolution[0] // 2  # max error in pixel
    # Use mutable to be modified by signal handler
    should_exit = [False]

    # Stop the robot on ctrl+c and exit the script
    def ctrl_c(signum, frame):
        print("STOP")
        should_exit[0] = True

    signal.signal(signal.SIGINT, ctrl_c)
    last_time = time.time()
    last_time_update = time.time()
    pbar = tqdm(total=n_seconds)
    n_total = 0  # For monitoring control frequency

    while time.time() - start_time < n_seconds and not should_exit[0]:
        # Display progress bar
        if time.time() - last_time_update > 1:
            pbar.update(int(time.time() - last_time_update))
            last_time_update = time.time()

        # Output of image processing
        turn_percent, centroids = out_queue.get()

        # Compute the error to the center of the line
        # We want the line to be in the middle of the image
        # Here we use the farthest centroids
        # TODO: try with the mean of the centroids to reduce noise
        error = (x_center - centroids[-1, 0]) / max_error_px

        # Represent line curve as a number in [0, 1]
        # h = 0 -> straight line
        # h = 1 -> sharp turn
        h = np.clip(turn_percent / 100.0, 0, 1)
        # Update moving mean
        mean_h += ALPHA * (h - mean_h)

        # We are using the moving mean (which is less noisy)
        # for line curve estimation
        h = mean_h
        # Reduce max speed if it is a sharp turn
        v_max = h * MAX_SPEED_SHARP_TURN + (1 - h) * MAX_SPEED_STRAIGHT_LINE
        # Different Kp depending on the line curve
        Kp = h * Kp_turn + (1 - h) * Kp_line

        # Represent error as a number in [0, 1]
        # t = 0 -> no error, we are perfectly on the line
        # t = 1 -> maximal error
        t = np.clip(error / float(ERROR_MAX), 0, 1)
        # Reduce speed if we have a high error
        speed_order = t * MIN_SPEED + (1 - t) * v_max

        errorD = error - last_error
        # Update derivative error
        last_error = error

        # PID Control
        # TODO: add dt in the equation
        dt = time.time() - last_time
        u_angle = Kp * error + Kd * errorD + Ki * errorI
        # Update integral error
        errorI += error
        last_time = time.time()

        angle_order = theta_init - u_angle
        angle_order = np.clip(angle_order, THETA_MIN, THETA_MAX).astype(int)

        # Send orders to Arduino
        sendOrders(speed_order, angle_order)
        n_total += 1

    # SEND STOP ORDER at the end
    forceStop()
    # Make sure STOP order is sent
    time.sleep(0.2)
    pbar.close()
    print("Main loop: {:.2f} Hz".format(n_total / (time.time() - start_time)))


if __name__ == '__main__':
    try:
        # Open serial port (for communication with Arduino)
        serial_port = get_serial_ports()[0]
        serial_file = serial.Serial(port=serial_port, baudrate=BAUDRATE, timeout=0, writeTimeout=0)
    except Exception as e:
        raise e

    # Initialize communication with Arduino
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

    # Image processing queue, output centroids
    out_queue = queue.Queue()
    condition_lock = threading.Lock()
    exit_condition = threading.Condition(condition_lock)

    print("Starting Image Processing Thread")
    # It starts 2 threads:
    #  - one for retrieving images from camera
    #  - one for processing the images
    image_thread = ImageProcessingThread(Viewer(out_queue, CAMERA_RESOLUTION, debug=False, fps=FPS), exit_condition)
    image_thread.start()
    # Wait for camera warmup
    time.sleep(1)

    print("Starting Control Thread")
    main_control(out_queue, resolution=CAMERA_RESOLUTION, n_seconds=N_SECONDS)

    print("Exiting...")
    with exit_condition:
        exit_condition.notify_all()

    image_thread.join()
