"""
Compute time to process an image
It processes the images N_ITER times and print statistics
"""
from __future__ import print_function, with_statement, division

import time
import argparse

import cv2
import numpy as np

from constants import NUM_OUTPUT, MAX_WIDTH, MAX_HEIGHT
from .utils import loadLabels, loadNetwork, predict

parser = argparse.ArgumentParser(description='Benchmark line detection algorithm')
parser.add_argument('-n', '--num_iterations', help='Number of iteration', default=1, type=int)
parser.add_argument('-w', '--weights', help='Saved weights', default="cnn_model_tmp.pth", type=str)
parser.add_argument('--model_type', help='Model type: cnn', default="cnn", type=str, choices=['cnn', 'custom'])
args = parser.parse_args()

N_ITER = args.num_iterations

model = loadNetwork(args.weights, NUM_OUTPUT, args.model_type)
model.cpu()

time_deltas = []
for i in range(N_ITER):
    image = np.random.randint(255) * np.ones((MAX_WIDTH, MAX_HEIGHT, 3), dtype=np.uint8)
    start_time = time.time()
    x, y = predict(model, image)
    time_deltas.append(time.time() - start_time)

time_deltas = np.array(time_deltas)
print("Total time: {:.6f}s".format(time_deltas.sum()))
print("Mean time: {:.4f}ms".format(1000 * time_deltas.mean()))
print("Std time: {:.4f}ms".format(1000 * time_deltas.std()))
print("Median time: {:.4f}ms".format(1000 * np.median(time_deltas)))
