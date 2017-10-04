from __future__ import print_function, with_statement, division, unicode_literals

import argparse
from common import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serial Listener')
    parser.add_argument('-i', '--input_file', help='Input File', default="", type=str)

    args = parser.parse_args()

    if args.input_file != "":
        try:
            serial_file = open(args.input_file, 'rb')
        except Exception as e:
            raise e
    else:
        # Automatically find the right port (ex: '/dev/ttyACM0')
        try:
            serial_port = get_serial_ports()[0]
            print("Connecting to {}".format(serial_port))
            serial_file = serial.Serial(port=serial_port, baudrate=BAUDRATE, timeout=0, writeTimeout=0)
        except Exception as e:
            raise e

    while True:
        bytes_array = bytearray(serial_file.read(1))
        if not bytes_array:
            time.sleep(1)
            continue
            # break
        byte = bytes_array[0]
        try:
            order = Order(byte)
        except ValueError as e:
            continue
        decodeOrder(serial_file, byte, debug=True)
