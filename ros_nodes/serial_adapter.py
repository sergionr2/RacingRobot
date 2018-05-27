import threading
import time

import numpy as np
import rospy
from std_msgs.msg import Int16, Int8
from robust_serial import write_order, Order
from robust_serial.threads import CommandThread, ListenerThread
from robust_serial.utils import open_serial_port, CustomQueue

from constants import BAUDRATE, N_MESSAGES_ALLOWED, COMMAND_QUEUE_SIZE, THETA_MIN, THETA_MAX


def servoCallback(data):
    servo_order = data.data
    servo_order = np.clip(servo_order, 0, 180)
    command_queue.put((Order.SERVO, servo_order))


def motorCallback(data):
    speed = data.data
    command_queue.put((Order.MOTOR, speed))


def listener():
    rospy.init_node('serial_adapter', anonymous=True)
    # Declare the Subscriber to motors orders
    rospy.Subscriber("arduino/servo", Int16, servoCallback, queue_size=2)
    rospy.Subscriber("arduino/motor", Int8, motorCallback, queue_size=2)
    rospy.spin()


def forceStop():
    """
    Stop The car
    """
    command_queue.clear()
    n_received_semaphore.release()
    n_received_semaphore.release()
    command_queue.put((Order.MOTOR, 0))
    command_queue.put((Order.SERVO, int((THETA_MIN + THETA_MAX) / 2)))


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
        print("Waiting for arduino...")
        write_order(serial_file, Order.HELLO)
        bytes_array = bytearray(serial_file.read(1))
        if not bytes_array:
            time.sleep(2)
            continue
        byte = bytes_array[0]
        if byte in [Order.HELLO.value, Order.ALREADY_CONNECTED.value]:
            is_connected = True

    print("Connected to Arduino")

    # Create Command queue for sending orders
    command_queue = CustomQueue(COMMAND_QUEUE_SIZE)
    n_received_semaphore = threading.Semaphore(N_MESSAGES_ALLOWED)
    # Lock for accessing serial file (to avoid reading and writing at the same time)
    serial_lock = threading.Lock()
    # Event to notify threads that they should terminate
    exit_event = threading.Event()

    print("Starting Communication Threads")
    # Threads for arduino communication
    threads = [CommandThread(serial_file, command_queue, exit_event, n_received_semaphore, serial_lock),
               ListenerThread(serial_file, exit_event, n_received_semaphore, serial_lock)]
    for thread in threads:
        thread.start()

    try:
        listener()
    except rospy.ROSInterruptException:
        pass

    # End the threads
    exit_event.set()
    n_received_semaphore.release()

    print("Exiting...")
    for thread in threads:
        thread.join()
