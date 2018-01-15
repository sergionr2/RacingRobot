"""
Compute time to process an image
It processes the images N_ITER times and print statistics
"""
from __future__ import print_function, with_statement, division

import time
import argparse

import cv2
import numpy as np

from opencv.image_processing import processImage
from opencv.c_extension import fastProcessImage

parser = argparse.ArgumentParser(description='Benchmark line detection algorithm')
parser.add_argument('-i', '--input_image', help='Input Image', default="", type=str, required=False)
parser.add_argument('-n', '--num_iterations', help='Number of iteration', default=1, type=int, required=False)

args = parser.parse_args()

N_ITER = args.num_iterations

image = cv2.imread(args.input_image).astype(np.float32)
# image = 155 * np.ones((240, 320, 3), dtype=np.uint8)
time_deltas = []
for i in range(N_ITER):
    start_time = time.time()
    # turn_percent, centroids = processImage(image, debug=False, regions=None, interactive=False)
    turn_percent, centroids = fastProcessImage(image)
    time_deltas.append(time.time() - start_time)
    # print(centroids)
    # print(turn_percent)

time_deltas = np.array(time_deltas)
print("Total time: {:.6f}s".format(time_deltas.sum()))
print("Mean time: {:.4f}ms".format(1000 * time_deltas.mean()))
print("Std time: {:.4f}ms".format(1000 * time_deltas.std()))
print("Median time: {:.4f}ms".format(1000 * np.median(time_deltas)))
