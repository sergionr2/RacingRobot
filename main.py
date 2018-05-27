"""
Main script for autonomous mode
It launches all the thread and does the PID control
use export OMP_NUM_THREADS=2 to improve performances
"""
from __future__ import division, print_function

import logging
import signal
import threading
import time
from datetime import datetime

# Python 2/3 support
try:
    import queue
except ImportError:
    import Queue as queue

import numpy as np
from tqdm import tqdm
from robust_serial import write_order, Order
from robust_serial.threads import CommandThread, ListenerThread
from robust_serial.utils import open_serial_port, CustomQueue

from picam.image_analyser import ImageProcessingThread, Viewer
from constants import THETA_MIN, THETA_MAX, ERROR_MAX, MAX_SPEED_SHARP_TURN, MAX_SPEED_STRAIGHT_LINE, \
    MIN_SPEED, Kp_turn, Kp_line, Kd, Ki, FPS, N_SECONDS, ALPHA, CAMERA_RESOLUTION, \
    BAUDRATE, N_MESSAGES_ALLOWED, COMMAND_QUEUE_SIZE

emptyException = queue.Empty
fullException = queue.Full

# Logging
log = logging.getLogger('racing_robot')
log.setLevel(logging.DEBUG)

# Formatter for logger
formatter = logging.Formatter('%(asctime)s -- %(levelname)s - %(message)s')

# Create file handler which logs even debug messages
fh = logging.FileHandler('logs/{}.log'.format(datetime.now().strftime("%y-%m-%d_%Hh%M_%S")))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
log.addHandler(fh)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
log.addHandler(ch)


def forceStop(command_queue, n_received_semaphore):
    """
    Stop The car
    :param command_queue: (CustomQueue) Queue for sending orders to the Arduino
    :param n_received_semaphore: (threading.Semaphore) Semaphore to regulate orders sent to the Arduino
    """
    command_queue.clear()
    n_received_semaphore.release()
    n_received_semaphore.release()
    command_queue.put((Order.MOTOR, 0))
    command_queue.put((Order.SERVO, int((THETA_MIN + THETA_MAX) / 2)))


def mainControl(command_queue, n_received_semaphore, out_queue, resolution, n_seconds=5):
    """
    :param command_queue: (CustomQueue) Queue for sending orders to the Arduino
    :param n_received_semaphore: (threading.Semaphore) Semaphore to regulate orders sent to the Arduino
    :param out_queue: (Queue) Output of image processing
    :param resolution: (int, int) Camera Resolution
    :param n_seconds: (int) number of seconds to keep this script alive
    """
    # Moving mean for line curve estimation
    mean_h = 0
    start_time = time.time()
    error, error_d, error_i = 0, 0, 0
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
        log.info("STOP")
        should_exit[0] = True

    signal.signal(signal.SIGINT, ctrl_c)
    last_time = time.time()
    last_time_update = time.time()
    pbar = tqdm(total=n_seconds)
    # Number of time the command queue was full
    n_full = 0
    n_total = 0

    log.debug("Entering in the control loop")

    while time.time() - start_time < n_seconds and not should_exit[0]:
        # Display progress bar
        if time.time() - last_time_update > 1:
            pbar.update(int(time.time() - last_time_update))
            last_time_update = time.time()

        # Output of image processing
        turn_percent, x_pred = out_queue.get()

        # Compute the error to the center of the line
        # We want the line to be in the middle of the image
        error = (x_center - x_pred) / max_error_px

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

        error_d = error - last_error
        # Update derivative error
        last_error = error

        # PID Control
        dt = time.time() - last_time
        u_angle = Kp * error + Kd * (error_d / dt) + Ki * (error_i * dt)
        # Update integral error
        error_i += error
        last_time = time.time()

        angle_order = theta_init - u_angle
        angle_order = np.clip(angle_order, THETA_MIN, THETA_MAX).astype(int)

        # Send orders to Arduino
        try:
            command_queue.put_nowait((Order.MOTOR, int(speed_order)))
            command_queue.put_nowait((Order.SERVO, angle_order))
        except fullException:
            n_full += 1
            # print("Command queue is full")
        n_total += 1

        # Logging
        log.debug("Error={:.2f} error_d={:.2f} error_i={:.2f}".format(error, error_d, error_i))
        log.debug("Turn percent={:.2f} x_pred={:.2f}".format(turn_percent, x_pred))
        log.debug("v_max={:.2f} mean_h={:.2f}".format(v_max, mean_h))
        log.debug("speed={:.2f} angle={:.2f}".format(speed_order, angle_order))

    # SEND STOP ORDER at the end
    forceStop(command_queue, n_received_semaphore)
    # Make sure STOP order is sent
    time.sleep(0.2)
    pbar.close()
    log.info("{:.2f}% of time the command queue was full".format(100 * n_full / n_total))
    log.info("Main loop: {:.2f} Hz".format((n_total - n_full) / (time.time() - start_time)))


if __name__ == '__main__':
    serial_file = None
    try:
        # Open serial port (for communication with Arduino)
        serial_file = open_serial_port(baudrate=BAUDRATE)
    except Exception as e:
        raise e

    is_connected = False
    # Initialize communication with Arduino
    while not is_connected:
        log.info("Waiting for arduino...")
        write_order(serial_file, Order.HELLO)
        bytes_array = bytearray(serial_file.read(1))
        if not bytes_array:
            time.sleep(2)
            continue
        byte = bytes_array[0]
        if byte in [Order.HELLO.value, Order.ALREADY_CONNECTED.value]:
            is_connected = True

    log.info("Connected to Arduino")

    # Image processing queue, output centroids
    out_queue = queue.Queue()
    condition_lock = threading.Lock()
    exit_condition = threading.Condition(condition_lock)

    log.info("Starting Image Processing Thread")
    # It starts 2 threads:
    #  - one for retrieving images from camera
    #  - one for processing the images
    image_thread = ImageProcessingThread(Viewer(out_queue, CAMERA_RESOLUTION, debug=False, fps=FPS), exit_condition)
    # Wait for camera warmup
    time.sleep(1)

    # Event to notify threads that they should terminate
    exit_event = threading.Event()

    # Communication with the Arduino
    # Create Command queue for sending orders
    command_queue = CustomQueue(COMMAND_QUEUE_SIZE)
    n_received_semaphore = threading.Semaphore(N_MESSAGES_ALLOWED)
    # Lock for accessing serial file (to avoid reading and writing at the same time)
    serial_lock = threading.Lock()

    log.info("Starting Communication Threads")
    # Threads for arduino communication
    threads = [CommandThread(serial_file, command_queue, exit_event, n_received_semaphore, serial_lock),
               ListenerThread(serial_file, exit_event, n_received_semaphore, serial_lock)]
    for thread in threads:
        thread.start()

    log.info("Starting Control Thread")
    mainControl(command_queue, n_received_semaphore, out_queue,
                resolution=CAMERA_RESOLUTION, n_seconds=N_SECONDS)

    # End the threads
    exit_event.set()
    n_received_semaphore.release()

    log.info("Exiting...")
    with exit_condition:
        exit_condition.notify_all()

    for thread in threads:
        thread.join()
