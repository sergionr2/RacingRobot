from __future__ import print_function, with_statement, division

import argparse
import time

import zmq
import picamera

import common
from common import *

try:
    import queue

except ImportError:
    import Queue as queue

emptyException = queue.Empty
fullException = queue.Full

# Listen to port 5556
port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:{}".format(port))

parser = argparse.ArgumentParser(description='Teleoperation server')
parser.add_argument('-v','--video_file', help='Video filename',  default="", type=str)
args = parser.parse_args()

record_video = args.video_file != ""
if record_video:
	print("Recording a video to {}".format(args.video_file))


try:
	serial_port = get_serial_ports()[0]
	serial_file = serial.Serial(port=serial_port, baudrate=BAUDRATE, timeout=0, writeTimeout=0)
except Exception as e:
	raise e

# Wait until we are connected to the arduino
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

threads = [CommandThread(serial_file, command_queue),
		   ListenerThread(serial_file)]
for t in threads:
	t.start()

print("Connected to Arduino, waiting for client...")
socket.send(b'1')
print("Connected To Client")
i = 0

with picamera.PiCamera() as camera:
	camera.resolution = (640//2, 480//2)
	if record_video:
		camera.start_recording("{}.h264".format(args.video_file))

	while True:
		control_speed, angle_order = socket.recv_json()
		print(control_speed, angle_order)
		try:
			if i%2 == 0:
				i = 1
				common.command_queue.put_nowait((Order.MOTOR, control_speed))
				common.command_queue.put_nowait((Order.SERVO, angle_order))
			else:
				i = 2
				common.command_queue.put_nowait((Order.SERVO, angle_order))
				common.command_queue.put_nowait((Order.MOTOR, control_speed))
		except fullException:
			print("Queue full")

		if control_speed == -999:
			socket.close()
			break
	if record_video:
		camera.stop_recording()

print("Sending STOP order...")
# SEND STOP ORDER at the end
common.resetCommandQueue()
n_received_semaphore.release()
common.command_queue.put((Order.MOTOR, 0))
# Make sure STOP order is sent
time.sleep(0.2)

common.exit_signal = True
n_received_semaphore.release()
print("EXIT")

for t in threads:
	t.join()
