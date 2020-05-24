#ifndef PARAMETERS_H
#define PARAMETERS_H

#define SERIAL_BAUD 115200
#define MOTOR_PIN 3
#define DIRECTION_PIN 4
#define SERVOMOTOR_PIN 6
// Initial angle of the servomotor
#define INITIAL_THETA 110
// Min and max values for motors
#define THETA_MIN 60
#define THETA_MAX 150
#define SPEED_MAX 100
// If DEBUG is set to true, the arduino will send back all the received messages
#define DEBUG false
// Set speed and angle to zero if no messages where received
// after 500ms
#define TIMEOUT 500

#endif
