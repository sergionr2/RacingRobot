"""
Main script for processing an image:
it extracts the different ROI, detect the line and estimate line curve
"""
from __future__ import print_function, with_statement, division, absolute_import

import argparse

import cv2
import numpy as np

from constants import REF_ANGLE, MAX_ANGLE, REGIONS, EXIT_KEYS, WIDTH, HEIGHT
from train import preprocessImage, loadVanillaNet, loadPytorchNetwork


# Either load network with pytorch or with numpy
# pred_fn = loadPytorchNetwork()
pred_fn = loadVanillaNet()


def mouseCallback(event, x, y, flags, centers):
    """
    Callback in interactive (annotation) mode
    """
    # Save the position of the mouse on left click
    if event == cv2.EVENT_LBUTTONDOWN:
        centers[0] = (x, y)


def processImage(image, debug=False, regions=None, interactive=False):
    """
    :param image: (rgb image)
    :param debug: (bool)
    :param regions: [[int]]
    :param interactive: (bool)
    :return:(float, numpy array)
    """
    im_width = image.shape[1]
    if regions is None:
        # Regions of interest
        regions = REGIONS

    centroids = np.zeros((len(regions), 2), dtype=int)
    errors = [False for _ in regions]
    exit_interactive = False  # For interactive mode

    # Efficient implementation
    if not debug:
        # Batch Prediction
        pred_imgs = []
        # Preprocess each region of interest
        for idx, r in enumerate(regions):
            im_cropped = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            # Preprocess the image: scaling and normalization
            pred_imgs.append(preprocessImage(im_cropped, WIDTH, HEIGHT))
        # Predict where is the center of the line using the trained network
        # and scale the output
        centroids[:, 0] = pred_fn(np.array(pred_imgs, dtype=np.float32))[:, 0] * im_width
        # Add left margin
        centroids[:, 0] += regions[:, 0]
        # Add top margin + set y_center to the the middle height
        centroids[:, 1] = regions[:, 3] // 2 + regions[:, 1]
    else:
        for idx, r in enumerate(regions):
            center = {}
            margin_left, margin_top, _, _ = r
            im_cropped = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

            im_cropped_tmp = im_cropped.copy()
            im_width = im_cropped_tmp.shape[1]
            pred_img = preprocessImage(im_cropped, WIDTH, HEIGHT)
            # Predict where is the center of the line using the trained network
            x_center = int(pred_fn([pred_img])[0] * im_width)
            y_center = im_cropped_tmp.shape[0] // 2

            if debug:
                # Draw prediction and true center
                cv2.circle(im_cropped_tmp, (x_center, y_center), radius=10,
                           color=(0, 0, 255), thickness=2, lineType=8, shift=0)
                cv2.imshow('crop_pred{}'.format(idx), im_cropped_tmp)

                # display line: y = height // 2
                h = im_cropped.shape[0] // 2
                cv2.line(im_cropped, (0, h), (1000, h), (0, 0, 255), 2)
                cv2.imshow('crop{}'.format(idx), im_cropped)
                # Labeling mode
                if interactive:
                    # Retrieve mouse click position
                    cv2.setMouseCallback('crop{}'.format(idx), mouseCallback, center)
                    key = cv2.waitKey(0) & 0xff
                    exit_interactive = key in EXIT_KEYS

            if debug and interactive:
                if center.get(0):
                    cx, cy = center[0]
                else:
                    cx, cy = 0, 0
                    errors[idx] = True
            else:
                cx, cy = x_center, y_center

            centroids[idx] = np.array([cx + margin_left, cy + margin_top])

    # Linear Regression to fit a line
    # It estimates the line curve
    x = centroids[:, 0]
    y = centroids[:, 1]
    # Case x = cst, m = 0
    if len(np.unique(x)) == 1:
        pts = centroids[:2, :]
        turn_percent = 0
    else:
        # Linear regression using least squares method
        # x = m*y + b -> y = 1/m * x - b/m if m != 0
        A = np.vstack([y, np.ones(len(y))]).T
        m, b = np.linalg.lstsq(A, x)[0]

        if debug:
            # Points for plotting the line
            y = np.array([0, image.shape[0]], dtype=int)
            pts = (np.vstack([m * y + b, y]).T).astype(int)
        # Compute the angle between the reference and the fitted line
        track_angle = np.arctan(1 / m)
        diff_angle = abs(REF_ANGLE) - abs(track_angle)
        # Estimation of the line curvature
        turn_percent = (diff_angle / MAX_ANGLE) * 100

    if debug and len(centroids) > 1:
        a, b = pts
    else:
        a, b = (0, 0), (0, 0)

    if debug:
        if all(errors):
            cv2.imshow('result', image)
        else:
            for cx, cy in centroids:
                cv2.circle(image, (cx, cy), radius=10, color=(0, 0, 255), thickness=1, lineType=8, shift=0)
                cv2.line(image, tuple(a), tuple(b), color=(100, 100, 0), thickness=2, lineType=8)
            cv2.line(image, (im_width // 2, 0), (im_width // 2, image.shape[0]),
                     color=(100, 0, 0), thickness=2, lineType=8)
            cv2.imshow('result', image)

    if interactive:
        return centroids, errors, exit_interactive
    return turn_percent, centroids


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Line Detection')
    parser.add_argument('-i', '--input_image', help='Input Image', default="", type=str)

    args = parser.parse_args()
    if args.input_image != "":
        img = cv2.imread(args.input_image)
        turn_percent, centroids = processImage(img, debug=True)
        if cv2.waitKey(0) & 0xff in EXIT_KEYS:
            cv2.destroyAllWindows()
            exit()
