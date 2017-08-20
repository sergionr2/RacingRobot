#include <iostream>
#include <string>
#include <algorithm> // For strtolower
#include <exception>
#include <fstream>
#include <cstdint>
#include <bitset>//Binary representation
#include <unistd.h>
#include <SerialStream.h>  // libserial-dev
#include "serial_reader.h"
#include "order.h"

using namespace LibSerial;
using namespace std;

SerialPort serial_port(PORT);

int main(int argc, char const *argv[])
{
	const unsigned int msTimeout = TIMEOUT_MS;
	Order currentOrder;
	serial_port.Open(SerialPort::BAUD_115200, SerialPort::CHAR_SIZE_8,
									 SerialPort::PARITY_NONE, SerialPort::STOP_BITS_1,
									 SerialPort::FLOW_CONTROL_NONE);

	for (size_t i = 0; i <= MAX_N_ORDER; i++)
	{
		// while (!serial_port.IsDataAvailable()){}
		try
		{
			currentOrder = readOrder(msTimeout);
			//  ReadTimeout
		}
		catch(exception& e)
		{
			std::cout << "Timeout!" << endl;
			serial_port.Close();
			return 0;
		}

		switch (currentOrder)
		{
			case HELLO:
			{
				cout << "HELLO" << endl;
				break;
			}
			case SERVO:
			{
				int16_t angle = readTwoBytesIntFromSerial();
				cout << "SERVO: angle=" << angle << endl;
				break;
			}
			case MOTOR:
			{
				int speed = readOneByteIntFromSerial();
				cout << "MOTOR: speed=" << speed << endl;
				break;
			}
			case ALREADY_CONNECTED:
			{
				cout << "ALREADY_CONNECTED" << endl;
				break;
			}
			case ERROR:
			{
				int16_t errorCode = readTwoBytesIntFromSerial();
				cout << "ERROR " << errorCode << endl;
				break;
			}
			case RECEIVED:
			{
				cout << "RECEIVED" << endl;
				break;
			}
			case STOP:
			{
				cout << "stop!" << endl;
				break;
			}
			default:
			{
				bitset<8> b(currentOrder);
				cout << "Unknown command:" << b << endl;
			}
		}
	}

	serial_port.Close();
  // WEIRD STUFF DO NOT REMOVE LINE FOR NOW
  stoul("1");
  return 0;
}

/**
 * Read two bytes (16 bits) from the serial and convert it to an unsigned int
 * @return the decoded unsigned int
 */
uint16_t readUnsignedIntFromSerial()
{
  SerialPort::DataBuffer charBuff;
  serial_port.Read(charBuff, 2);
  return (uint16_t)(charBuff[0] & 0xff) | (uint16_t)(charBuff[1] << 8);
}
/**
 * Read two bytes from the serial and convert it to an int
 * @return the decoded int
 */
int16_t readTwoBytesIntFromSerial()
{
  SerialPort::DataBuffer charBuff;
  serial_port.Read(charBuff, 2);
  return (int16_t)(charBuff[0] & 0xff) | (int16_t)(charBuff[1] << 8 & 0xff00);
}

/**
 * Read four bytes from the serial and convert it to an long
 * @return the decoded int
 */
int32_t readFourBytesIntFromSerial()
{
  SerialPort::DataBuffer charBuff;
  serial_port.Read(charBuff, 4);
  return (int8_t)(charBuff[0] & 0xff) | (int8_t)(charBuff[1] << 8 & 0xff00) | (int8_t)(charBuff[2] << 16 & 0xff0000) | (int8_t)(charBuff[3] << 24 & 0xff000000);
}

/**
 * Read one byte from the serial and convert it to an int
 * @return the decoded int
 */
int8_t readOneByteIntFromSerial()
{
  SerialPort::DataBuffer charBuff;
  serial_port.Read(charBuff, 1);
  // We have to cast it to keep the sign
  return (int8_t) static_cast<signed char>(charBuff[0]);
}

/**
 * Read one byte from the serial and cast it to an Order
 * @return the order received
 */
Order readOrder(const unsigned int msTimeout)
{
  SerialPort::DataBuffer buffer;
  serial_port.Read(buffer, 1, msTimeout);
  return (Order) buffer[0];
}
