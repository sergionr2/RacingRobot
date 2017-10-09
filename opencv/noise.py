"""
Add noise to an input image to emulate camera shake
"""
import math
import random

import cv2
import numpy as np


def rotateImage(image, phi, theta, psi):
    """
    Rotate an image
    :param image: (cv2 image object)
    :param phi: (float)
    :parma theta: (float)
    :param psi: (float)
    :return: (cv2 image object)
    """
    # Height, Width, Channels
    h, w, c = image.shape
    F = np.float32([[300, 0, w / 2.], [0, 300, h / 2.], [0, 0, 1]])
    R = rotMatrix([phi, theta, psi])
    T = [[0], [0], [1]]
    T = np.dot(R, T)
    R[0][2] = T[0][0]
    R[1][2] = T[1][0]
    R[2][2] = T[2][0]
    M = np.dot(F, np.linalg.inv(np.dot(F, R)))
    out = cv2.warpPerspective(image, M, (w, h))
    return out


def rotMatrix(theta):
    """
    Create a rotation tensor
    :param theta: [float]
    :return: (numpy tensor)
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


if __name__ == '__main__':

    img = cv2.imread("noise_demo_image.jpg")

    while True:
        noisedImage = rotateImage(img, random.random() * 0.02 - 0.01, random.random() * 0.02 - 0.01,
                                  random.random() * 0.02 - 0.01)  # 5 degrees
        cv2.imshow("img", noisedImage)
        cv2.waitKey()
