import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from constants import MAX_WIDTH, MAX_HEIGHT

# Transform Parameters
y = 90
a = 0.75
delta = (MAX_HEIGHT - y) * a

height, width = 500, 320
# Orignal and transformed keypoints
pts1 = np.float32(
    [[delta, y],
     [MAX_WIDTH - delta, y],
     [0, MAX_HEIGHT],
     [MAX_WIDTH, MAX_HEIGHT]])

pts2 = np.float32(
    [[0, 0],
     [width, 0],
     [0, height],
     [width, height]])

# Translation Matrix
tx, ty = 300, 500
T = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

new_height, new_width = height + ty, int(width * 1.5) + tx

# calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(pts1, pts2)


def imshow(im, y=None, delta=None, name=""):
    plt.figure(name)
    # BGR to RGB
    plt.imshow(im[:, :, ::-1])
    if y is not None:
        plt.plot([0, delta], [MAX_HEIGHT, y])
        plt.plot([MAX_WIDTH, MAX_WIDTH - delta], [MAX_HEIGHT, y])
        plt.plot([delta, MAX_WIDTH - delta], [y, y])
    plt.grid(True)


def showTransform(image, y, delta):
    im = image.copy()
    for (cx, cy) in pts1:
        cv2.circle(im, (int(cx), int(cy)), 8, (0, 255, 0), -1)
    imshow(im, y, delta, name="transform")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform image to have a top down view')
    parser.add_argument('-i', '--input_image', help='Input image', type=str, required=True)
    args = parser.parse_args()

    image = cv2.imread(args.input_image)
    assert image is not None, "Could not read image"
    orignal_image = image.copy()
    warp = cv2.warpPerspective(orignal_image, np.dot(T, M), (new_width, new_height))
    imshow(image, name="original")
    showTransform(image, y, delta)
    imshow(warp, name="warped")
    plt.show()
