#include <Arduino.h>
#include <Servo.h>

#include "order.h"
#include "slave.h"
#include "parameters.h"

bool isConnected = false; ///< True if the connection with the master is available
int8_t motorSpeed = 0;
int16_t servoPosition = INITIAL_THETA;
unsigned long int lastMillis = 0;
const int cycleDuration = 5; // ms
Servo servomotor;

void setup()
{
  // Init Serial
  Serial.begin(SERIAL_BAUD);

  // Init Motor
  pinMode(MOTOR_PIN , OUTPUT);
  pinMode(DIRECTION_PIN , OUTPUT);
  // Stop the car
  stop();

  // Init Servo
  servomotor.attach(SERVOMOTOR_PIN);
  // Order between 0 and 180
  servomotor.write(INITIAL_THETA);

  // Wait until the arduino is connected to master
  while(!isConnected)
  {
    sendOrder(HELLO);
    waitForBytes(1, 1000);
    getMessageFromSerial();
  }
  lastMillis = millis();
}

void loop()
{
  getMessageFromSerial();
  if(millis() - lastMillis > cycleDuration)
  {
    lastMillis = millis();
    cycle(); //run this function after every cycleDuration
  }
}

void cycle()
{
  // PID Control -> send orders to motors
  servomotor.write(constrain(servoPosition, THETA_MIN, THETA_MAX));
  motorSpeed = constrain(motorSpeed, -SPEED_MAX, SPEED_MAX);
  // Send motor speed order
  if (motorSpeed > 0)
  {
    digitalWrite(DIRECTION_PIN , LOW);
  }
  else
  {
    digitalWrite(DIRECTION_PIN , HIGH);
  }
  analogWrite(MOTOR_PIN , convertOrderToPWM(float(motorSpeed)));

}

void stop()
{
  analogWrite(MOTOR_PIN , 0);
  digitalWrite(DIRECTION_PIN , LOW);
}

int convertOrderToPWM(float speedOrder)
{
  // TODO: compensate the non-linear dependency speed = f(PWM_Value)
  return (int) round(abs(speedOrder)*(255./100.));
}

void getMessageFromSerial()
{
  while(Serial.available() > 0)
  {
    //The first byte received is the instruction
    Order orderReceived = readOrder();

    //Commands for initConnection begin
    if(orderReceived == HELLO)
    {
      //If the cards haven't say hello, check the connection
      if (!isConnected)
      {
        isConnected = true;
        sendOrder(HELLO);
      }
      else
      {
        //If we are already connected do not send "hello" to avoid infinite loop
        sendOrder(ALREADY_CONNECTED);
      }
    }
    else if(orderReceived == ALREADY_CONNECTED)
    {
      isConnected = true;
    }
    //Commands for initConnection end
    else // Commands after initConnection
    {
      switch(orderReceived)
      {
        case STOP:
        {
          motorSpeed = 0;
          stop();
          if (DEBUG)
          {
            sendOrder(STOP);
          }
          break;
        }
        case SERVO:
        {
          servoPosition = readTwoBytesIntFromSerial();
          if (DEBUG)
          {
            sendOrder(SERVO);
            sendTwoBytesInt(servoPosition);
          }
          break;
        }
        case MOTOR:
        {
          // between -100 and 100
          motorSpeed = readOneByteIntFromSerial();
          if (MOTOR)
          {
            sendOrder(MOTOR);
            sendOneByteInt(motorSpeed);
          }
          break;
        }
  			// Unknown order
  			default:
          sendOrder(ERROR);
          sendTwoBytesInt(404);
  				return;
      }
      sendOrder(RECEIVED); // Confirm the reception
    }
  }
}


Order readOrder()
{
	return (Order) Serial.read();
}

void waitForBytes(int nbOfBytes, unsigned long timeout)
{
	unsigned long startTime = millis();
	//Wait for incoming bytes or exit if timeout
	while ((Serial.available() < nbOfBytes) && (millis() - startTime < timeout)){}
}

void readSignedBytes(int8_t* buffer, size_t n)
{
	size_t i = 0;
	int c;
	while (i < n)
	{
		c = Serial.read();
		if (c < 0) break;
		*buffer++ = (int8_t)c;
		i++;
	}
}

// NOTE : Serial.readBytes is SLOW
// this one is much faster, but has no timeout
void readBytes(uint8_t* buffer, size_t n)
{
	size_t i = 0;
	int c;
	while (i < n)
	{
		c = Serial.read();
		if (c < 0) break;
		*buffer++ = (uint8_t)c;// buffer[i] = (uint8_t)c;
		i++;
	}
}

int8_t readOneByteIntFromSerial()
{
	waitForBytes(1, 100);//Wait for 1 byte with a timeout of 100 ms
  return (int8_t) Serial.read();
}

int16_t readTwoBytesIntFromSerial()
{
  int8_t buffer[2];
	waitForBytes(2, 100);//Wait for 2 bytes with a timeout of 100 ms
	readSignedBytes(buffer, 2);
  return (int16_t)(buffer[0] & 0xff) | (int16_t)(buffer[1] << 8 & 0xff00);
}

long readFourBytesIntFromSerial()
{
  int8_t buffer[4];
	waitForBytes(4, 200);//Wait for 4 bytes with a timeout of 200 ms
	readSignedBytes(buffer, 4);
  return (long)(buffer[0] & 0xff) | (long)(buffer[1] << 8 & 0xff00) | (long)(buffer[2] << 16 & 0xff0000) | (long)(buffer[3] << 24 & 0xff000000);
}

void sendOrder(enum Order myOrder)
{
	uint8_t* Order = (uint8_t*) &myOrder;
  Serial.write(Order, sizeof(uint8_t));
}

/**
 * Send a int of one byte (between -127 and 127)
 * @param myInt an int of one byte
 */
void sendOneByteInt(int8_t myInt)
{
  Serial.write(myInt);
}

/**
 * Send a two bytes signed int via the serial
 * @param nb the number to send (max: (2**16/2 -1) = 32767)
 */
void sendTwoBytesInt(int nb)
{
	int8_t buffer[2] = {(int8_t) (nb & 0xff), (int8_t) (nb >> 8)};
  Serial.write((uint8_t*)&buffer, 2*sizeof(int8_t));
}

/**
 * Send a four bytes signed int (long) via the serial
 * @param nb the number to send (−2,147,483,647, +2,147,483,647)
 */
void sendFourBytesInt(long nb)
{
	int8_t buffer[4] = {(int8_t) (nb & 0xff), (int8_t) (nb >> 8 & 0xff), (int8_t) (nb >> 16 & 0xff), (int8_t) (nb >> 24 & 0xff)};
  Serial.write((uint8_t*)&buffer, 4*sizeof(int8_t));
}
