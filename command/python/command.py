from __future__ import print_function, with_statement, division

import argparse
from common import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serial Listener')
    parser.add_argument('-o', '--output_file', help='Output File', default="", type=str)

    args = parser.parse_args()

    if args.output_file != "":
        try:
            serial_file = open(args.output_file, 'wb')
        except Exception as e:
            raise e
    else:
        # Automatically find the right port (ex: '/dev/ttyACM0')
        try:
            serial_port = get_serial_ports()[0]
            serial_file = serial.Serial(port=serial_port, baudrate=BAUDRATE, timeout=0, writeTimeout=0)
        except Exception as e:
            raise e

    for order in Order:
        sendOrder(serial_file, order.value)
        if order == Order.HELLO:
            pass
        elif order == Order.SERVO:
            angle = -3500
            writeTwoBytesInt(serial_file, angle)
        elif order == Order.MOTOR:
            speed = 100
            writeOneByteInt(serial_file, speed)
        elif order == Order.ALREADY_CONNECTED:
            pass
        elif order == Order.ERROR:
            code_error = 404
            writeTwoBytesInt(serial_file, code_error)
        elif order == Order.RECEIVED:
            pass
        elif order == Order.STOP:
            pass
        else:
            pass
