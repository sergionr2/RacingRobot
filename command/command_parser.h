#ifndef COMMAND_PARSER_H
#define COMMAND_PARSER_H
#include "order.h"

#define PORT "/dev/ttyACM0"

int getIntFromUserInput(std::string infoMessage);
unsigned int getUnsignedIntFromUserInput(std::string infoMessage);
long getLongIntFromUserInput(std::string infoMessage);

void sendOneOrder(enum Order myOrder);

void sendOrder(enum Order myOrder);
void sendOneByteInt(int8_t nb);
void sendTwoBytesInt(int16_t nb);
void sendFourBytesInt(int32_t nb);
void sendTwoBytesUnsignedInt(uint16_t nb);

#endif
