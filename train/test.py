"""
Test the trained model on images
"""
from __future__ import print_function, division, absolute_import

import argparse
import os
import sys

import cv2
import numpy as np

from constants import RIGHT_KEY, LEFT_KEY, EXIT_KEYS, ROI, NUM_OUTPUT, TARGET_POINT, MODEL_TYPE, WEIGHTS_PTH
from constants import MAX_WIDTH, MAX_HEIGHT
from path_planning.bezier_curve import calcBezierPath, computeControlPoints, bezier, calcTrajectory
from path_planning.stanley_controller import stanleyControl, State, calcTargetIndex
from image_processing.warp_image import warpImage, transformPoints
from .utils import loadLabels, loadNetwork, predict, computeMSE

parser = argparse.ArgumentParser(description='Test a line detector')
parser.add_argument('-i', '--input_video', help='Input Video', default="", type=str)
parser.add_argument('-f', '--folders', help='Dataset folders', nargs='+', default=[""], type=str)
parser.add_argument('-w', '--weights', help='Saved weights', default=WEIGHTS_PTH, type=str)
parser.add_argument('--model_type', help='Model type: {cnn, custom}', default=MODEL_TYPE, type=str,
                    choices=['cnn', 'custom'])
parser.add_argument('--no-display', action='store_true', default=False, help='Compute only mse')
parser.add_argument('--no-mse', action='store_true', default=False, help='Do not compute mse')

args = parser.parse_args()

assert args.folders[0] != "" or args.input_video != "", "You must specify a video or dataset for testing"

video = None
if args.input_video != "":  # pragma: no cover
    assert os.path.isfile(args.input_video), "Invalid path to input video"
    image_zero_index = cv2.CAP_PROP_POS_FRAMES
    frame_count = cv2.CAP_PROP_FRAME_COUNT
    video = cv2.VideoCapture(args.input_video)

model = loadNetwork(args.weights, NUM_OUTPUT, args.model_type)

labels, train_labels, val_labels, test_labels = {}, {}, {}, {}
images = None

# Load images from a folder
if video is None:
    if os.path.isfile("{}/labels.json".format(args.folders[0])):
        train_labels, val_labels, test_labels, labels = loadLabels(args.folders)
        if False:
            images = list(labels.keys())
    if images is None:
        images = []
        for folder in args.folders:
            tmp_images = ["{}/{}".format(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
            tmp_images.sort(key=lambda name: int(name.split('.jpg')[0].split('/')[-1]))
            images += tmp_images

    idx_val = set(val_labels.keys())
    idx_test = set(test_labels.keys())
    n_frames = len(images)
    current_idx = 0
else: # pragma: no cover
    # Load video
    if not video.isOpened():  # pragma: no cover
        print("Error opening video, check your opencv version (you may need to compile it from source)")
        sys.exit(1)
    current_idx = video.get(image_zero_index)
    n_frames = int(video.get(frame_count))

print("{} frames".format(n_frames))

if n_frames <= 0:  # pragma: no cover
    print("Not enough frame, check your path")
    sys.exit(1)

if len(train_labels) > 0 and not args.no_mse:
    computeMSE(model, train_labels, val_labels, test_labels, batchsize=16)
    if args.no_display:
        sys.exit(0)

while True:  # pragma: no cover
    if video is not None:
        while True:
            flag, image = video.read()
            if flag:
                break
            else:
                # The next frame is not ready, so we try to read it again
                video.set(image_zero_index, current_idx - 1)
                cv2.waitKey(1000)
                continue
        text = ""
        name = str(current_idx)
    else:
        path = images[current_idx]
        image = cv2.imread(path)
        # image = cv2.flip(image, 1)
        # Image from train/validation/test set ?
        text = "train"
        if path in idx_val:
            text = "val"
        elif path in idx_test:
            text = "test"

    x, y = predict(model, image)
    points = transformPoints(x, y).astype(np.int32)

    # print(current_idx)
    # Compute bezier path
    control_points = computeControlPoints(x, y, add_current_pos=True)
    target = bezier(TARGET_POINT, control_points).astype(np.int32)
    path = calcBezierPath(control_points).astype(np.int32)

    orignal_image = image.copy()
    cv2.putText(image, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

    true_labels = None
    if video is None and labels.get(images[current_idx]) is not None:
        true_labels = np.array(labels[images[current_idx]])

    for i in range(len(path) - 1):
        cv2.line(image, (path[i, 0], path[i, 1]), (path[i + 1, 0], path[i + 1, 1]),
                 color=(0, 0, int(0.8 * 255)), thickness=3)
    # Show Target point
    cv2.circle(image, tuple(target), radius=10, color=(0, 0, int(0.9 * 255)),
               thickness=1, lineType=8, shift=0)

    # Draw prediction and label
    for i in range(len(x) - 1):
        cv2.line(image, (x[i], y[i]), (x[i + 1], y[i + 1]), color=(176, 114, 76),
                 thickness=3)
        if true_labels is not None:
            cv2.line(image, (true_labels[i, 0], true_labels[i, 1]), (true_labels[i + 1, 0], true_labels[i + 1, 1]),
                     color=(104, 168, 85), thickness=3)
    cv2.imshow('Prediction', image)

    # Draw prediction on warped image
    warped_image = warpImage(orignal_image)
    for i in range(len(points)):
        cv2.circle(warped_image, (points[i, 0], points[i, 1]), radius=50, color=(int(0.9 * 255), 0, 0),
                   thickness=10, lineType=8, shift=0)
    for i in range(len(x) - 1):
        cv2.line(warped_image, (points[i, 0], points[i, 1]), (points[i + 1, 0], points[i + 1, 1]), color=(176, 114, 76),
                 thickness=10)


    path = calcBezierPath(points, n_points=10).astype(np.int32)

    cp = transformPoints(x, y)
    cp[:, 1] *= -1
    current_pos = transformPoints([MAX_WIDTH/2], [MAX_HEIGHT])[0]
    # TODO: compute v in the pixel space
    state = State(x=current_pos[0], y=-current_pos[1], yaw=np.radians(90.0), v=10)
    cx, cy, cyaw, ck = calcTrajectory(cp, n_points=10)
    target_idx, _ = calcTargetIndex(state, cx, cy)
    target = (int(cx[target_idx]), - int(cy[target_idx]))

    delta, _, cross_track_error = stanleyControl(state, cx, cy, cyaw, target_idx)
    # print(delta, cross_track_error)

    # Target
    cv2.circle(warped_image, target, radius=50, color=(0, 0, int(0.9 * 255)),
               thickness=10, lineType=8, shift=0)

    # Draw bezier curve
    for i in range(len(path) - 1):
        cv2.line(warped_image, (path[i, 0], path[i, 1]), (path[i + 1, 0], path[i + 1, 1]),
                 color=(0, 0, int(0.8 * 255)), thickness=10)

    warped_image = cv2.resize(warped_image, (warped_image.shape[1]//2, warped_image.shape[0]//2), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Warped image', warped_image)
    # r = ROI
    # im_cropped = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # cv2.imshow('Crop', im_cropped)

    key = cv2.waitKey(0) & 0xff
    if key in EXIT_KEYS:
        cv2.destroyAllWindows()
        break
    elif key in [LEFT_KEY, RIGHT_KEY]:
        current_idx += 1 if key == RIGHT_KEY else -1
        current_idx = np.clip(current_idx, 0, n_frames - 1)
    if video is not None:
        video.set(image_zero_index, current_idx)

if video is not None:  # pragma: no cover
    video.release()
