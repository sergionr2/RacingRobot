#ifndef SLAVE_H
#define SLAVE_H

/*!
 * \file slave.h
 * \authors Antonin RAFFIN
 * \date June 2017
 * \brief Code to communicate with the master and send orders to motors
 */


void cycle();
void stop();
int convertOrderToPWM(float speedOrder);

/*!
 * \brief Read one byte from the serial and cast it to an Order
 * \return the order received
 */
Order readOrder();

/*!
 * \brief Wait until there are enough bytes in the buffer
 * \param nbOfBytes the number of bytes
 * \param timeout (ms) The timeout, time after which we release the lock
 * even if there are not enough bytes
 */
void waitForBytes(int nbOfBytes, unsigned long timeout);

/*!
 * \brief Read signed bytes and put them in a buffer
 * \param buffer an array of bytes
 * \param n number of bytes to read
 */
void readSignedBytes(int8_t* buffer, size_t n);

/*!
* \brief Read signed bytes and put them in a buffer
* \param buffer an array of unsigned bytes
* \param n number of bytes to read
*/
void readBytes(uint8_t* buffer, size_t n);

/*!
 * \brief Read one byte from the serial and convert it to an int
 * \return the decoded int
 */
int8_t readOneByteIntFromSerial();

/*!
 * \brief Read two bytes from the serial and convert it to an int
 * \return the decoded int
 */
int16_t readTwoBytesIntFromSerial();


/*!
 * \brief Read four bytes from the serial and convert it to an long
 * \return the decoded int
 */
long readFourBytesIntFromSerial();

/*!
 * \brief Send one order (one byte) to the other arduino
 * \param myOrder type of order
 * \param debugging whether we should send it to the SerialDebugObject
 */
void sendOrder(enum Order myOrder);

/*!
 * \brief Send an int of one byte (between -127 and 127)
 * \param myInt an int of one byte
 * \param debugging whether we should send it to the SerialDebugObject
 */
void sendOneByteInt(int8_t myInt);

/*!
 * \brief Send a two bytes signed int via the serial
 * \param nb the number to send (max: (2**16/2 -1) = 32767)
 * \param debugging whether we should send it to the SerialDebugObject
 */
void sendTwoBytesInt(int nb);

/*!
 * \brief Send a four bytes signed int (long) via the serial
 * \param nb the number to send (−2,147,483,647, +2,147,483,647)
 * \param debugging whether we should send it to the SerialDebugObject
 */
void sendFourBytesInt(long nb);

/*!
 * \brief Listen the serial and decode the message received
 */
void getMessageFromSerial();

#endif
