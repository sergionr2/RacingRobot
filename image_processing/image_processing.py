"""
Main script for processing an image:
it preprocesses the image, detects the line, estimate line curve
and do the path planning
"""
from __future__ import print_function, with_statement, division, absolute_import

import argparse

import cv2
import numpy as np

from constants import REF_ANGLE, MAX_ANGLE, EXIT_KEYS, WEIGHTS_PTH, NUM_OUTPUT, MODEL_TYPE, TARGET_POINT
from train import loadNetwork, predict
from path_planning.bezier_curve import computeControlPoints, bezier

# Load trained model
model = loadNetwork(WEIGHTS_PTH, NUM_OUTPUT, MODEL_TYPE)


def processImage(image, debug=False):
    """
    :param image: (bgr image)
    :param debug: (bool)
    :return:(float, float)
    """
    x, y = predict(model, image)
    if debug:
        return x, y

    # Compute bezier path and target point
    control_points = computeControlPoints(x, y, add_current_pos=True)
    target = bezier(TARGET_POINT, control_points)

    # Linear Regression to fit a line
    # It estimates the line curve

    # Case x = cst, m = 0
    if len(np.unique(x)) == 1:
        turn_percent = 0
    else:
        # Linear regression using least squares method
        # x = m*y + b -> y = 1/m * x - b/m if m != 0
        A = np.vstack([y, np.ones(len(y))]).T
        m, b = np.linalg.lstsq(A, x)[0]

        # Compute the angle between the reference and the fitted line
        track_angle = np.arctan(1 / m)
        diff_angle = abs(REF_ANGLE) - abs(track_angle)
        # Estimation of the line curvature
        turn_percent = (diff_angle / MAX_ANGLE) * 100
    return turn_percent, target[0]
