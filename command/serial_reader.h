#ifndef SERIAL_READER_H
#define SERIAL_READER_H
#include "order.h"

#define PORT "/dev/ttyACM0"
#define MAX_N_ORDER 1000
#define TIMEOUT_MS 30000 // 30s

Order readOrder(const unsigned int msTimeout);
int8_t readOneByteIntFromSerial();
int16_t readTwoBytesIntFromSerial();
uint16_t readUnsignedIntFromSerial();
int32_t readFourBytesIntFromSerial();

#endif
