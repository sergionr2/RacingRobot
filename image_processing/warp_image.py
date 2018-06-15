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
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

new_height, new_width = int(height * 1) + ty + 600, int(width * 1.2) + tx

# calculate the perspective transform matrix
transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
complete_transform = np.dot(translation_matrix, transform_matrix)

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

def transformPoints(x, y):
    points = []
    for i in range(len(x)):
        point = transformPoint(np.array([x[i], y[i], 1]))
        scale = point[2]
        point = point / scale
        points.append(point[:2])
    return np.array(points)


def transformPoint(point):
    """
    :param points: (numpy array)
    :return: (numpy array)
    """
    return np.matmul(complete_transform, point)


def warpImage(image):
    # TODO: transform only points
    return cv2.warpPerspective(image, complete_transform, (new_width, new_height))

if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser(description='Transform image to have a top down view')
    parser.add_argument('-i', '--input_image', help='Input image', type=str, required=True)
    parser.add_argument('--no-display', action='store_true', default=False, help='Do not display plots (for tests)')
    args = parser.parse_args()

    image = cv2.imread(args.input_image)
    assert image is not None, "Could not read image"
    orignal_image = image.copy()
    warp = cv2.warpPerspective(orignal_image, np.dot(translation_matrix, transform_matrix), (new_width, new_height))
    if not args.no_display:
        imshow(image, name="original")
        showTransform(image)
        imshow(warp, name="warped")
        plt.show()
