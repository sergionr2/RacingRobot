from __future__ import print_function, with_statement, division

import time
import argparse

import cv2
import numpy as np

from opencv.image_processing import processImage

N_ITER = 5000

parser = argparse.ArgumentParser(description='Benchmark line detection algorithm')
parser.add_argument('-i', '--input_image', help='Input Image', default="", type=str, required=True)
args = parser.parse_args()


image = cv2.imread(args.input_image)
time_deltas = []
for i in range(N_ITER):
    start_time = time.time()
    turn_percent, centroids = processImage(image, debug=False, regions=None, interactive=False)
    time_deltas.append(time.time() - start_time)
    # print(centroids)

time_deltas = np.array(time_deltas)
print("Total time: {:.6f}s".format(time_deltas.sum()))
print("Mean time: {:.6f}s".format(time_deltas.mean()))
print("Std time: {:.6f}s".format(time_deltas.std()))
print("Median time: {:.6f}s".format(np.median(time_deltas)))
