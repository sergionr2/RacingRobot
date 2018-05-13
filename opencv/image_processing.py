"""
Main script for processing an image:
it extracts the different ROI, detect the line and estimate line curve
"""
from __future__ import print_function, with_statement, division, absolute_import

import argparse

import cv2
import numpy as np

from constants import REF_ANGLE, MAX_ANGLE, EXIT_KEYS, WEIGHTS_PTH, ROI, NUM_OUTPUT, Y_TARGET
from train import loadNetwork, predict

# Load trained model
model = loadNetwork(WEIGHTS_PTH, NUM_OUTPUT)


def processImage(image, debug=False):
    """
    :param image: (bgr image)
    :param debug: (bool)
    :return:(float, numpy array)
    """
    x, y = predict(model, image)
    if debug:
        return x, y

    # if y[1] <= Y_TARGET:
    #     y1, y2 = y[1], y[2]
    #     x1, x2 = x[1], x[2]
    # else:
    #     y1, y2 = y[0], y[1]
    #     x1, x2 = x[0], x[1]
    #
    # if y2 == y1:
    #     # TODO: improve this particular case
    #     x_pred = x[1]
    # else:
    #     # x = a * y + b
    #     a = (x2 - x1) / (y2 - y1)
    #     b = x1 - a * y1
    #     x_pred = a * Y_TARGET + b

    x_pred = x[1]
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
    return turn_percent, x_pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Line Detection')
    parser.add_argument('-i', '--input_image', help='Input Image', default="", type=str)

    args = parser.parse_args()
    if args.input_image != "":
        img = cv2.imread(args.input_image)
        x, y = processImage(img, debug=True)
        for i in range(len(x) - 1):
            cv2.line(img, (x[i], y[i]), (x[i + 1], y[i + 1]), color=(176, 114, 76),
                     thickness=3)
        cv2.imshow('Prediction', img)
        if cv2.waitKey(0) & 0xff in EXIT_KEYS:
            cv2.destroyAllWindows()
            exit()
