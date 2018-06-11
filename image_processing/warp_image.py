import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from constants import MAX_WIDTH, MAX_HEIGHT

# Transform Parameters
height, width = 1200, 800
# Orignal and transformed keypoints
pts1 = np.float32(
    [[103, 93],
     [222, 95],
     [6, 130],
     [310, 128]])

pts2 = np.float32(
    [[0, 0],
     [width, 0],
     [0, height],
     [width, height]])

# Translation Matrix
tx, ty = 0, 0
T = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

new_height, new_width = int(height * 1) + ty + 300, int(width * 1.2) + tx

# calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(pts1, pts2)


def imshow(im, name=""):  # pragma: no cover
    plt.figure(name)
    # BGR to RGB
    plt.imshow(im[:, :, ::-1])
    plt.grid(True)


def showTransform(image):  # pragma: no cover
    im = image.copy()
    for (cx, cy) in pts1:
        cv2.circle(im, (int(cx), int(cy)), 8, (0, 255, 0), -1)
    imshow(im, name="transform")


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser(description='Transform image to have a top down view')
    parser.add_argument('-i', '--input_image', help='Input image', type=str, required=True)
    parser.add_argument('--no-display', action='store_true', default=False, help='Do not display plots (for tests)')
    args = parser.parse_args()

    image = cv2.imread(args.input_image)
    assert image is not None, "Could not read image"
    orignal_image = image.copy()
    warp = cv2.warpPerspective(orignal_image, np.dot(T, M), (new_width, new_height))
    if not args.no_display:
        imshow(image, name="original")
        showTransform(image)
        imshow(warp, name="warped")
        plt.show()
