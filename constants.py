"""
File containing all the constants used in the different files
"""
from __future__ import print_function, division, absolute_import

import numpy as np

# Main Constants
CAMERA_RESOLUTION = (640 // 2, 480 // 2)
# https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
# mode 4, larger FoV but FPS <= 40
CAMERA_MODE = 4
# Regions of interest
MAX_WIDTH = CAMERA_RESOLUTION[0]
MAX_HEIGHT = CAMERA_RESOLUTION[1]
# r = [margin_left, margin_top, width, height]
ROI = [0, 75, MAX_WIDTH, MAX_HEIGHT - 75]

# Image Analyser
RECORD_VIDEO = True  # Record video to debug folder

# Training
FACTOR = 4  # Resize factor
INPUT_WIDTH = ROI[2] // FACTOR
INPUT_HEIGHT = ROI[3] // FACTOR
N_CHANNELS = 3
SPLIT_SEED = 42  # For train/val/test split
MODEL_TYPE = "custom"  # Network architecture {cnn or custom}
WEIGHTS_PTH = MODEL_TYPE + "_model.pth"  # Path to the trained model
NUM_OUTPUT = 6  # Predict 3 points -> 6 outputs

# Direction and speed
THETA_MIN = 70  # value in [0, 255] sent to the servo
THETA_MAX = 150
MAX_SPEED_STRAIGHT_LINE = 35  # order between 0 and 100
MAX_SPEED_SHARP_TURN = 30
MIN_SPEED = 20

# Path Planning
# TODO: compute curvature of bezier path to regulate speed
# see https://stackoverflow.com/questions/46762955
TARGET_POINT = 0.8  # between [0, 1] position on the bezier curve

# Stanley Steering Control params
# Control gain (compromise between smooth control
# and precise track following)
K_STANLEY_CONTROL = 1
CAR_LENGTH = 1.5  # [m] Wheel base of vehicle
MAX_STEERING_ANGLE = np.radians(25.0)  # [rad]
# Radius of curvature
MIN_RADIUS = 5  # Below this value, we consider to be in a sharp turn
MAX_RADIUS = 40  # Above this value, we consider to be in a straight line

# PID Control
Kp_turn = 50
Kp_line = 50
Kd = 4
Ki = 0
ALPHA = 1  # alpha of the moving mean for the turn coefficient
# Main Program
FPS = 40
N_SECONDS = 3000  # number of seconds before exiting the program
BAUDRATE = 115200  # Communication with the Arduino
# Number of messages we can send to the Arduino without receiving an acknowledgment
N_MESSAGES_ALLOWED = 3
COMMAND_QUEUE_SIZE = 2

# Image Processing
# Straight line angle
REF_ANGLE = - np.pi / 2
# Max turn angle
MAX_ANGLE = np.pi / 4  # 2 * np.pi / 3

# Teleoperation
TELEOP_PORT = "5556"
RASPBERRY_IP = "192.168.12.252"


# Arrow keys
UP_KEY = 82
DOWN_KEY = 84
RIGHT_KEY = 83
LEFT_KEY = 81
ENTER_KEY = 10
SPACE_KEY = 32
EXIT_KEYS = [113, 27]  # Escape and q
S_KEY = 115  # S key
