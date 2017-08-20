#include <iostream>
#include <string>
#include <algorithm> // For strtolower
#include <exception>
#include <fstream>
#include <cstdint>
#include <bitset>//Binary representation
#include <unistd.h>
#include "command_parser.h"
#include "order.h"

using namespace std;

fstream serialFile;

int main(int argc, char const *argv[])
{
	string defaultFileName = "/dev/ttyACM0";
	string serialFileName = "";
	string cmd;// the command name
	bool exitPrompt = false;
	bool validCommand = true;
	cout << "Enter the name of the serial file (default: /dev/ttyACM0 )" << endl;
  getline(cin, serialFileName);

	if(serialFileName.empty())
	{
		cout << "Using default serial file : " << defaultFileName << endl;
		serialFileName = defaultFileName;
	}
	// Open the serial file (read/write mode)
	// ios::app Append mode. All output to that file to be appended to the end.
  serialFile.open(serialFileName, ios::in|ios::out|ios::app|ios::binary);
	// If we couldn't open the output file stream for writing
	if (!serialFile)
	{
		// Print an error and exit
		cerr << "Uh oh, serialFile could not be opened for writing!" << endl;
		exit(1);
	}

	while (!exitPrompt)
	{
		cout << "=========" << endl;
		cout << "Commands: hello | motor | servo | stop | exit " << endl;
		cout << "=========" << endl;
		if(cmd != "")
		{
			if(!validCommand)
			{
				cmd += " Invalid command!";
				validCommand = true;
			}
			cout << "Last command: " << cmd << endl;
		}
		cout << "Please enter a command" << endl;
	  cin >> cmd;
	  // To lower case
	  transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);
		if (cmd == "motor")
		{
			int speed = getIntFromUserInput("speed (between -100 and 100) ?");
			sendOrder(MOTOR);
			sendOneByteInt(speed);
			cmd += " " + to_string(speed);
		}
		else if (cmd == "servo")
		{
			int angle = getIntFromUserInput("angle (between  0 and 180) ?");
			sendOrder(SERVO);
			sendTwoBytesInt(angle);
			cmd += " " + to_string(angle);
		}
		else if (cmd == "hello")
		{
			sendOrder(HELLO);
		}
		else if (cmd == "stop")
		{
			sendOrder(STOP);
		}
		else if (cmd == "exit")
		{
			exitPrompt = true;
		}
		else
		{
			validCommand = false;
			cout << endl << "Unknown command ! " << endl << endl;
		}
		flush(serialFile);// Write data in the buffer to the output file
		cout << "\033[2J\033[1;1H";//Clear the terminal
	}
	serialFile.close();
  return 0;
}

/**
 * Send one order (one byte) to the other arduino
 * @param myOrder type of order
 */
void sendOrder(enum Order myOrder)
{
	uint8_t* Order = (uint8_t*) &myOrder;
  serialFile.write((char *)Order, sizeof(uint8_t));
}

/**
 * Send a int of one byte
 * @param myOrder type of order
 */
void sendOneByteInt(int8_t myInt)
{
	int8_t* oneByte = (int8_t*) &myInt;
  serialFile.write((char *)oneByte, sizeof(int8_t));
}


/**
 * Send a two bytes signed int via the serial
 * @param nb the number to send
 */
void sendTwoBytesInt(int16_t nb)
{
	int8_t buffer[2] = {(int8_t) (nb & 0xff), (int8_t) (nb >> 8)};
	serialFile.write((char *)buffer, 2*sizeof(int8_t));
}

/**
 * Send a four bytes signed int (long) via the serial
 * @param nb the number to send (−2,147,483,647, +2,147,483,647)
 */
void sendFourBytesInt(int32_t nb)
{
	int8_t buffer[4] = {(int8_t) (nb & 0xff), (int8_t) (nb >> 8 & 0xff), (int8_t) (nb >> 16 & 0xff), (int8_t) (nb >> 24 & 0xff)};
  serialFile.write((char *)buffer, 4*sizeof(int8_t));
}

/**
 * Send a two bytes unsigned (max 2**16 -1) int via the serial
 * @param nb the number to send
 */
void sendTwoBytesUnsignedInt(uint16_t nb)
{
	uint8_t buffer[2] = {(uint8_t) (nb & 0xff), (uint8_t) (nb >> 8)};
	serialFile.write((char *)buffer, 2*sizeof(uint8_t));
}

/**
 * Ask the user to enter an integer
 * Prompt until the input is valid
 * @param  infoMessage The message displayed to the user
 * @return   the integer entered by the user
 */
int getIntFromUserInput(std::string infoMessage)
{
  bool isValid = false;
  int intParam;
  string param;
  while(!isValid)
  {
    cout << infoMessage << endl;// Ask the user for input
    cin >> param;// Store the string entered
    try
    {
      // Convert String to int
      intParam = stoi(param);
      isValid = true;
    }
    catch(exception& e)
    {
      isValid = false;
      cout << "Please enter a valid integer "<< endl << endl;
    }
  }
  return intParam;
}

/**
 * Ask the user to enter a float
 * Prompt until the input is valid
 * @param  infoMessage The message displayed to the user
 * @return   the integer entered by the user
 */
float getFloatFromUserInput(std::string infoMessage)
{
  bool isValid = false;
  float floatParam;
  string param;
  while(!isValid)
  {
    cout << infoMessage << endl;// Ask the user for input
    cin >> param;// Store the string entered
    try
    {
      // Convert String to float
      floatParam = stof(param);
      isValid = true;
    }
    catch(exception& e)
    {
      isValid = false;
      cout << "Please enter a valid float "<< endl << endl;
    }
  }
  return floatParam;
}

unsigned int getUnsignedIntFromUserInput(std::string infoMessage)
{
	bool isValid = false;
	unsigned int intParam;
	string param;
	while(!isValid)
	{
		cout << infoMessage << endl;// Ask the user for input
		cin >> param;// Store the string entered
		try
		{
			// Convert String to int
			intParam = stoul(param);
			isValid = true;
		}
		catch(exception& e)
		{
			isValid = false;
			cout << "Please enter a valid unsigned integer "<< endl << endl;
		}
	}
	return intParam;
}

/**
 * Ask the user to enter an long
 * Prompt until the input is valid
 * @param  infoMessage The message displayed to the user
 * @return   the integer entered by the user
 */
long getLongIntFromUserInput(std::string infoMessage)
{
  bool isValid = false;
  long intParam;
  string param;
  while(!isValid)
  {
    cout << infoMessage << endl;// Ask the user for input
    cin >> param;// Store the string entered
    try
    {
      // Convert String to long int
      intParam = stol(param);
      isValid = true;
    }
    catch(exception& e)
    {
      isValid = false;
      cout << "Please enter a valid long integer "<< endl << endl;
    }
  }
  return intParam;
}
