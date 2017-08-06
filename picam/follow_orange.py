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

THETA_MIN = 60
THETA_MAX = 130

def main(out_queue, MAX_CX=320, n_seconds=5):
    start_time = time.time()
    while time.time() - start_time < n_seconds:
        cx, cy, error = out_queue.get()
        if not error:
            print(cx, cy, error)
            t = cx / float(MAX_CX)
            print(t)
            angle_order = int(THETA_MIN  * t + THETA_MAX * (1 - t))
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

    resolution = (640, 480)
    out_queue = queue.Queue()
    condition_lock = threading.Lock()
    condition = threading.Condition(condition_lock)
    image_thread = ImageProcessingThread(Viewer(out_queue, resolution), condition)

    threads = [CommandThread(serial_file, command_queue),
               ListenerThread(serial_file), image_thread]
    for t in threads:
        t.start()

    main(out_queue, MAX_CX=resolution[0], n_seconds=1*30)

    common.exit_signal = True
    n_received_semaphore.release()

    print("EXIT")
    # End the thread
    with condition:
        condition.notify_all()

    for t in threads:
        t.join()
