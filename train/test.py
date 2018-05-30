"""
Test the trained model on images
"""
from __future__ import print_function, division, absolute_import

import argparse
import os

import cv2
import numpy as np

from constants import RIGHT_KEY, LEFT_KEY, EXIT_KEYS, ROI, NUM_OUTPUT, TARGET_POINT
from path_planning.bezier_curve import calcBezierPath, computeControlPoints, bezier
from .utils import loadLabels, loadNetwork, predict

parser = argparse.ArgumentParser(description='Test a line detector')
parser.add_argument('-f', '--folder', help='Training folder', type=str, required=True)
parser.add_argument('-w', '--weights', help='Saved weights', default="cnn_model_tmp.pth", type=str)
parser.add_argument('--model_type', help='Model type: {cnn, custom}', default="custom", type=str, choices=['cnn', 'custom'])

args = parser.parse_args()

current_idx = 0
model = loadNetwork(args.weights, NUM_OUTPUT, args.model_type)

train_labels, val_labels, test_labels, labels = loadLabels(args.folder)

images = list(labels.keys())
# TODO: add support for folders without labels
if True:
    images = [f for f in os.listdir(args.folder) if f.endswith('.jpg')]
    # labels = {}

images.sort(key=lambda name: int(name.split('.jpg')[0]))

idx_val = set(val_labels.keys())
idx_test = set(test_labels.keys())

# TODO: compute val and test error

while True:
    name = images[current_idx]
    image = cv2.imread('{}/{}'.format(args.folder, images[current_idx]))
    # image = cv2.flip(image, 1)
    # Image from train/validation/test set ?
    text = "train"
    if name in idx_val:
        text = "val"
    elif name in idx_test:
        text = "test"

    x, y = predict(model, image)
    # print(current_idx)
    # Compute bezier path
    control_points = computeControlPoints(x, y, add_current_pos=True)
    target = bezier(TARGET_POINT, control_points).astype(np.int32)
    path = calcBezierPath(control_points).astype(np.int32)

    cv2.putText(image, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

    if labels.get(images[current_idx]) is not None:
        true_labels = np.array(labels[images[current_idx]])
    else:
        true_labels = None

    for i in range(len(path) - 1):
        cv2.line(image, (path[i, 0], path[i, 1]), (path[i + 1, 0], path[i + 1, 1]), color=(0, 0, int(0.8 * 255)),
                         thickness=3)
    # Show Target point
    cv2.circle(image, tuple(target), radius=10, color=(0, 0, int(0.9 * 255)), thickness=1, lineType=8, shift=0)

    # Draw prediction and label
    for i in range(len(x) - 1):
        cv2.line(image, (x[i], y[i]), (x[i + 1], y[i + 1]), color=(176, 114, 76),
                 thickness=3)
        if true_labels is not None:
            cv2.line(image, (true_labels[i, 0], true_labels[i, 1]), (true_labels[i + 1, 0], true_labels[i + 1, 1]),
                     color=(104, 168, 85),
                     thickness=3)
    cv2.imshow('Prediction', image)

    # r = ROI
    # im_cropped = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # cv2.imshow('Crop', im_cropped)

    key = cv2.waitKey(0) & 0xff
    if key in EXIT_KEYS:
        cv2.destroyAllWindows()
        break
    elif key in [LEFT_KEY, RIGHT_KEY]:
        current_idx += 1 if key == RIGHT_KEY else -1
        current_idx = np.clip(current_idx, 0, len(images) - 1)
