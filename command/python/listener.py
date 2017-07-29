import argparse
from common import *

def decodeOrder(f, byte):
    """
    :param f: file handler or serial file
    :param byte: (int8_t)
    """
    order = Order(byte)
    if order == Order.HELLO:
        print("hello")
    elif order == Order.SERVO:
        angle = readTwoBytesInt(f)
        # Bit representation
        # print('{0:016b}'.format(angle))
        print("servo {}".format(angle))
    elif order == Order.MOTOR:
        speed = readOneByteInt(f)
        print("motor {}".format(speed))
    elif order == Order.ALREADY_CONNECTED:
        print("ALREADY_CONNECTED")
    elif order == Order.ERROR:
        code_error = readTwoBytesInt(f)
        print("Error {}".format(code_error))
    elif order == Order.RECEIVED:
        print("RECEIVED")
    elif order == Order.STOP:
        print("STOP")
    else:
        print("unknown order", byte)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Serial Listener')
    parser.add_argument('-i','--input_file', help='Input File',  default="", type=str)

    args = parser.parse_args()

    if args.input_file != "":
        try:
            serial_file = open(args.input_file, 'rb')
        except Exception as e:
            raise e
    else:
        # Automatically find the right port (ex: '/dev/ttyACM0')
        try:
            serial_file = get_serial_ports()[0]
            serial_port = serial.Serial(port=serial_file, baudrate=BAUDRATE, timeout=0, writeTimeout=0)
        except Exception as e:
            raise e

    while True:
       bytes_array = serial_file.read(1)
       if not bytes_array:
          break
       byte = bytes_array[0]
       decodeOrder(serial_file, byte)
