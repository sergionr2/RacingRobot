"""
Test the trained model on images
"""
from __future__ import print_function, division, absolute_import

import time
import argparse

import cv2
import numpy as np
import torch as th
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


from constants import UP_KEY, DOWN_KEY, RIGHT_KEY, LEFT_KEY, EXIT_KEYS, SPLIT_SEED
from .train import loadDataset, loadNetwork, loadPytorchNetwork
from .utils import computeMSE

parser = argparse.ArgumentParser(description='Test a line detector')
parser.add_argument('-f', '--folder', help='Training folder', default="", type=str, required=True)
parser.add_argument('-w', '--weights', help='Saved weights', default="", type=str, required=True)
parser.add_argument('--pytorch', action='store_true', default=False, help='Use pytorch model')
parser.add_argument('--no-data-augmentation', action='store_true', default=False, help='Disables data augmentation')

args = parser.parse_args()

augmented = not args.no_data_augmentation

# Load dataset
X, y_true, images = loadDataset(folder=args.folder, split=False, augmented=augmented)
indices = np.arange(len(y_true))
idx_train, idx_test = train_test_split(indices, test_size=0.4, random_state=SPLIT_SEED)
idx_val, idx_test = train_test_split(idx_test, test_size=0.5, random_state=SPLIT_SEED)

# Load trained model
pytorch = args.pytorch

if pytorch:
    pred_fn = loadPytorchNetwork(args.weights)
    start_time = time.time()
    y_test = pred_fn(Variable(th.from_numpy(X))).data.numpy()[:, 0]
    total_time = time.time() - start_time
    print("\nTime to predict: {:.2f}s | {:.5f} ms/image".format(total_time, 1000 * total_time / len(y_true)))
else:
    network, pred_fn = loadNetwork(args.weights)
    y_test = pred_fn(X)[:, 0]

# Compute Loss
computeMSE(y_test, y_true, [idx_train, idx_val, idx_test])

current_idx = 0

while True:
    name = images[current_idx]
    im = cv2.imread('{}/{}'.format(args.folder, images[current_idx]))
    # By convention, mirrored images are at the end
    if augmented and current_idx >= len(images) // 2:
        im = cv2.flip(im, 1)
    height, width, n_channels = im.shape

    # Image from train/validation/test set ?
    text = "train"
    if current_idx in idx_val:
        text = "val"
    elif current_idx in idx_test:
        text = "test"
    cv2.putText(im, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

    # x_center, y_center = map(int, name.split('_')[0].split('-'))
    x_true = int(y_true[current_idx] * width)
    x_center = int(y_test[current_idx] * width)
    x_center = np.clip(x_center, 0, width)
    y_center = height // 2
    print(current_idx, name, "error={}".format(abs(x_center - x_true)))

    # Draw prediction and true center
    cv2.circle(im, (x_center, y_center), radius=10, color=(0, 0, 255),
               thickness=2, lineType=8, shift=0)
    cv2.circle(im, (x_true, y_center), radius=10, color=(255, 0, 0),
               thickness=1, lineType=8, shift=0)
    cv2.imshow('Prediction', im)

    key = cv2.waitKey(0) & 0xff
    if key in EXIT_KEYS:
        cv2.destroyAllWindows()
        break
    elif key in [LEFT_KEY, RIGHT_KEY]:
        current_idx += 1 if key == RIGHT_KEY else -1
        current_idx = np.clip(current_idx, 0, len(images) - 1)
