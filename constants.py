"""
File containing all the constants used in the different files
"""
import numpy as np

# Main Constants
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
FPS = 60
N_SECONDS = 77  # number of seconds before exiting the program
ALPHA = 0.8  # alpha of the moving mean for the turn coefficient

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
