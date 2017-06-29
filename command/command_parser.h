#ifndef MAIN_H
#define MAIN_H
#include "order.h"

#define PORT "/dev/ttyACM0"

int getIntFromUserInput(std::string infoMessage);
unsigned int getUnsignedIntFromUserInput(std::string infoMessage);
long getLongIntFromUserInput(std::string infoMessage);

void sendOneOrder(enum Order myOrder);

void sendOrder(enum Order myOrder);
void sendOneByteInt(int nb);
void sendTwoBytesInt(int nb);
void sendFourBytesInt(long nb);
void sendTwoBytesUnsignedInt(unsigned int nb);

#endif
