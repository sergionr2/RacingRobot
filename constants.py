"""
File containing all the constants used in the different files
"""
import numpy as np

# Main Constants
CAMERA_RESOLUTION = (640 // 2, 480 // 2)
# Regions of interest
MAX_WIDTH = CAMERA_RESOLUTION[0]
# r = [margin_left, margin_top, width, height]
R0 = [0, 150, MAX_WIDTH, 50]
R1 = [0, 125, MAX_WIDTH, 50]
R2 = [0, 100, MAX_WIDTH, 50]
R3 = [0, 75, MAX_WIDTH, 50]
R4 = [0, 50, MAX_WIDTH, 50]
REGIONS = np.array([R1, R2, R3])
# Training
WIDTH, HEIGHT = 80, 20  # Shape of the resized input image fed to our model

THETA_MIN = 70  # value in [0, 255] sent to the servo
THETA_MAX = 150
ERROR_MAX = 1.0
MAX_SPEED_STRAIGHT_LINE = 50  # order between 0 and 100
MAX_SPEED_SHARP_TURN = 15
MIN_SPEED = 10
# PID Control
Kp_turn = 40
Kp_line = 35
Kd = 30
Ki = 0.0
ALPHA = 0.8  # alpha of the moving mean for the turn coefficient
# Main Program
FPS = 60
N_SECONDS = 77  # number of seconds before exiting the program

# Image Analyser
SAVE_EVERY = 1000  # Save every 1000 frame to debug folder

# Image Processing
# Straight line angle
REF_ANGLE = - np.pi / 2
# Max turn angle
MAX_ANGLE = np.pi / 4  # 2 * np.pi / 3


# Arrow keys
UP_KEY = 82
DOWN_KEY = 84
RIGHT_KEY = 83
LEFT_KEY = 81
ENTER_KEY = 10
SPACE_KEY = 32
EXIT_KEYS = [113, 27]  # Escape and q
S_KEY = 115  # Save key
