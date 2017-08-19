from __future__ import division, print_function

import time
import threading
try:
    import queue
except ImportError:
    import Queue as queue

import command.python.common as common
from command.python.common import *
from image_analyser import *

# THETA_MIN = 60
# THETA_MAX = 130
THETA_MIN = 0
THETA_MAX = 190

def main(out_queue, MAX_CX=320, n_seconds=5, roi=None):
    start_time = time.time()
    MIN_CX = 0
    if roi is not None:
    	margin_left, _, width, _ = roi
    	MAX_CX = width + margin_left
    	MIN_CX = margin_left
    while time.time() - start_time < n_seconds:
        cx, cy, error = out_queue.get()
        if not error:
            #print(cx, cy, error)
            t = (cx - MIN_CX) / MAX_CX
            t = 1 - t
            angle_order = int(THETA_MIN  * t + THETA_MAX * (1 - t))
            #print(t, angle_order)
            common.command_queue.put((Order.SERVO, angle_order))

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
    out_queue = queue.Queue()
    condition_lock = threading.Lock()
    condition = threading.Condition(condition_lock)
    image_thread = ImageProcessingThread(Viewer(out_queue, resolution, debug=False, fps=40), condition)

    threads = [CommandThread(serial_file, command_queue),
               ListenerThread(serial_file), image_thread]
    for t in threads:
        t.start()

    region_of_interest = [50, 150, 200, 50]
    main(out_queue, MAX_CX=resolution[0], n_seconds=1*30, roi=region_of_interest)

    common.exit_signal = True
    n_received_semaphore.release()

    print("EXIT")
    # End the thread
    with condition:
        condition.notify_all()

    for t in threads:
        t.join()
