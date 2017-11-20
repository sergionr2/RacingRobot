import rospy
from std_msgs.msg import Int16, Int8

import time
import threading

import serial
import numpy as np

import command.python.common as common

from command.python.common import is_connected, n_received_semaphore, command_queue, \
    CommandThread, ListenerThread, sendOrder, Order, get_serial_ports, BAUDRATE

def servoCallback(data):
    servo_order = data.data
    servo_order = np.clip(servo_order, 0, 180)
    common.command_queue.put((Order.SERVO, servo_order))

def motorCallback(data):
    speed = data.data
    common.command_queue.put((Order.MOTOR, speed))

def listener():
    rospy.init_node('serial_adapter', anonymous=True)
    # Declare the Subscriber to center_deviationturn
    rospy.Subscriber("arduino/servo", Int16, servoCallback, queue_size=2)
    rospy.Subscriber("arduino/motor", Int8, motorCallback, queue_size=2)
    rospy.spin()

def forceStop():
    # SEND STOP ORDER at the end
    common.resetCommandQueue()
    n_received_semaphore.release()
    n_received_semaphore.release()
    common.command_queue.put((Order.MOTOR, 0))
    common.command_queue.put((Order.SERVO, int(110)))


if __name__ == '__main__':
    try:
        serial_port = get_serial_ports()[0]
        serial_file = serial.Serial(port=serial_port, baudrate=BAUDRATE, timeout=0, writeTimeout=0)
    except Exception as e:
        raise e
    # Connect to the arduino
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

    exit_event = threading.Event()

    print("Starting Communication Threads")
    # Threads for arduino communication
    threads = [CommandThread(serial_file, command_queue, exit_event),
               ListenerThread(serial_file, exit_event)]
    for t in threads:
        t.start()

    try:
        listener()
    except rospy.ROSInterruptException:
        pass

    # End the threads
    exit_event.set()
    n_received_semaphore.release()

    print("Exiting...")
    for t in threads:
        t.join()
