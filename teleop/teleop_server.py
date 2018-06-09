from __future__ import print_function, with_statement, division

import argparse
import time
import threading

try:
    import queue
except ImportError:
    import Queue as queue

import zmq
try:
    import picamera
except ImportError:
    raise ImportError("Picamera package not found, you must run this code on the Raspberry Pi")

from robust_serial import write_order, Order
from robust_serial.threads import CommandThread, ListenerThread
from robust_serial.utils import open_serial_port, CustomQueue

from constants import CAMERA_RESOLUTION, CAMERA_MODE, TELEOP_PORT


emptyException = queue.Empty
fullException = queue.Full

# Listen to port 5556
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:{}".format(TELEOP_PORT))

parser = argparse.ArgumentParser(description='Teleoperation server')
parser.add_argument('-v', '--video_file', help='Video filename', default="", type=str)
args = parser.parse_args()

record_video = args.video_file != ""
if record_video:
    print("Recording a video to {}".format(args.video_file))

try:
    serial_file = open_serial_port(baudrate=115200)
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
command_queue = CustomQueue(2)
# Number of messages we can send to the Arduino without receiving an acknowledgment
n_messages_allowed = 3
n_received_semaphore = threading.Semaphore(n_messages_allowed)
# Lock for accessing serial file (to avoid reading and writing at the same time)
serial_lock = threading.Lock()

# Event to notify threads that they should terminate
exit_event = threading.Event()

print("Starting Communication Threads")
# Threads for arduino communication
threads = [CommandThread(serial_file, command_queue, exit_event, n_received_semaphore, serial_lock),
           ListenerThread(serial_file, exit_event, n_received_semaphore, serial_lock)]
for t in threads:
    t.start()

print("Waiting for client...")
socket.send(b'1')
print("Connected To Client")
i = 0

with picamera.PiCamera() as camera:
    camera.resolution = CAMERA_RESOLUTION
    camera.sensor_mode = CAMERA_MODE
    if record_video:
        camera.start_recording("{}.h264".format(args.video_file))

    while True:
        control_speed, angle_order = socket.recv_json()
        print("({}, {})".format(control_speed, angle_order))
        try:
            command_queue.put_nowait((Order.MOTOR, control_speed))
            command_queue.put_nowait((Order.SERVO, angle_order))
        except fullException:
            print("Queue full")

        if control_speed == -999:
            socket.close()
            break
    if record_video:
        camera.stop_recording()

print("Sending STOP order...")
# Stop the car at the end
command_queue.clear()
n_received_semaphore.release()
command_queue.put((Order.MOTOR, 0))
# Make sure STOP order is sent
time.sleep(0.2)
# End the threads
exit_event.set()
n_received_semaphore.release()
print("Exiting...")

for t in threads:
    t.join()
