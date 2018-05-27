// Code using Robust Arduino Serial Protocol: https://github.com/araffin/arduino-robust-serial

#ifndef ARDUINO_SLAVE_H
#define ARDUINO_SLAVE_H

/*!
 * \brief Send updated motors orders to the two motors (servomotor + motor)
 */
void update_motors_orders();

/*!
 * Stop the car (set the speed to 0)
 */
void stop();

/*!
 * \brief Convert a speed order (in percentage of max speed)
 * into a pwm order (between 0 and 255)
 * \param motor_speed speed order in percentage of max speed
 * \return the speed order in pwm
 */
int convert_to_pwm(float motor_speed);

/*!
 * \brief Read one byte from the serial and cast it to an Order
 * \return the order received
 */
Order read_order();

/*!
 * \brief Wait until there are enough bytes in the buffer
 * \param num_bytes the number of bytes
 * \param timeout (ms) The timeout, time after which we release the lock
 * even if there are not enough bytes
 */
void wait_for_bytes(int num_bytes, unsigned long timeout);

/*!
 * \brief Read signed bytes and put them in a buffer
 * \param buffer an array of bytes
 * \param n number of bytes to read
 */
void read_signed_bytes(int8_t* buffer, size_t n);

/*!
 * \brief Read one byte from a serial port and convert it to a 8 bits int
 * \return the decoded number
 */
int8_t read_i8();

/*!
 * \brief Read two bytes from a serial port and convert it to a 16 bits int
 * \return the decoded number
 */
int16_t read_i16();


/*!
 * \brief Read four bytes from a serial port and convert it to a 32 bits int
 * \return the decoded number
 */
int32_t read_i32();

/*!
 * \brief Send one order (one byte)
 * \param order type of order
 */
void write_order(enum Order order);

/*!
 * \brief Write one byte int to serial port (between -127 and 127)
 * \param num an int of one byte
 */
void write_i8(int8_t num);

/*!
 * \brief Send a two bytes signed int via the serial
 * \param num the number to send (max: (2**16/2 -1) = 32767)
 */
void write_i16(int16_t num);

/*!
 * \brief Send a four bytes signed int (long) via the serial
 * \param num the number to send (−2,147,483,647, +2,147,483,647)
 */
void write_i32(int32_t num);

/*!
 * \brief Listen the serial and decode the message received
 */
void get_messages_from_serial();

#endif
