"""
Compute time to process an image
It processes the images N_ITER times and print statistics
"""
from __future__ import print_function, with_statement, division

import time
import argparse

import cv2
import numpy as np
import torch as th
from torch import jit
from torch.jit import trace

from constants import NUM_OUTPUT, MAX_WIDTH, MAX_HEIGHT, INPUT_WIDTH, INPUT_HEIGHT, N_CHANNELS
from .utils import loadLabels, loadNetwork, predict, preprocessImage

parser = argparse.ArgumentParser(description='Benchmark line detection algorithm')
parser.add_argument('-n', '--num_iterations', help='Number of iteration', default=1, type=int)
parser.add_argument('-w', '--weights', help='Saved weights', default="custom_model_tmp.pth", type=str)
parser.add_argument('--model_type', help='Model type: cnn', default="custom", type=str, choices=['cnn', 'custom'])
parser.add_argument('--jit', action='store_true', default=False, help='Use JIT')

args = parser.parse_args()

N_ITER = args.num_iterations

model = loadNetwork(args.weights, NUM_OUTPUT, args.model_type)
model = model.to("cpu")
# 1st way to compile a function
# model = jit.compile(model.forward, nderivs=1, enabled=args.jit)

# 2nd way to use the JIT
# Give an example input to compile the model
# TODO: use real image for Benchmark
example_input = th.ones((1, N_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)).to(th.float)
model = trace(example_input)(model)

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
