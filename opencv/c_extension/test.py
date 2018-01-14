from __future__ import print_function, division, absolute_import

import time

import numpy as np
import cv2

from test_module import forward, setWeights, processImage, forward2

N_ITER = 5000
batchsize = 3
np.random.seed(2)


weights_npy='mlp_model.npz'

W, b = {}, {}
with np.load(weights_npy) as f:
    n_layers = len(f.files) // 2
    for i in range(len(f.files)):
        # print(f['arr_%d' % i].shape)
        if i % 2 == 1:
            b[i // 2] = f['arr_%d' % i].astype(np.float32)
        else:
            W[i // 2] = f['arr_%d' % i].astype(np.float32)

setWeights(W[0], b[0], W[1], b[1], W[2], b[2])



def loadVanillaNet(weights_npy='mlp_model.npz'):
    """
    Load a trained network and
    return the forward function in pure numpy
    :param weights_npy: (str) path to the numpy archive
    :return: (function) the neural net forward function
    """
    W, b = {}, {}
    with np.load(weights_npy) as f:
        print("Loading network")
        n_layers = len(f.files) // 2
        for i in range(len(f.files)):
            if i % 2 == 1:
                b[i // 2] = f['arr_%d' % i].astype(np.float32)
            else:
                W[i // 2] = f['arr_%d' % i].astype(np.float32)

    def relu(x):
        """
        Rectify activation function: f(x) = max(0, x)
        :param x: (numpy array)
        :return: (numpy array)
        """
        y = x.copy()
        y[y < 0] = 0
        return y

    def forward(X):
        """
        Forward pass of a fully-connected neural net
        with rectifier activation function
        :param X: (numpy tensor)
        :return: (numpy array)
        """
        a = X
        for i in range(n_layers):
            z = np.dot(a, W[i]) + b[i]
            a = relu(z)
        return a

    return forward

model_py = loadVanillaNet()

a = np.random.uniform(low=-1, high=1, size=(batchsize, 80, 20, 3)).reshape((batchsize, -1)).astype(np.float32)

# print(forward2(a))
# print(model_py(a))
# exit()

# print(forward(a))
# print(model_py(a))
# exit()
# start_time = time.time()
# forward2(N_ITER)
# print("Total cpp: {:.4f}s".format(time.time() - start_time))
image = cv2.imread("test_sun.jpg").astype(np.float32)

times = []
for _ in range(N_ITER):
    a = np.random.uniform(low=-1, high=1, size=(batchsize, 80, 20, 3)).reshape((batchsize, -1))
    a = a.astype(np.float32)
    start_time = time.time()
    # b = model_py(a)
    # b = forward(a)
    # b = forward2(a)
    r = processImage(image)
    times.append(time.time() - start_time)
    # print(r)
    # print(b)

print("Total: {:.4f}s".format(np.sum(times)))
print("Mean: {:.4f}ms".format(1000 * np.mean(times)))
print("Std: {:.4f}ms".format(1000 * np.std(times)))
print("Median: {:.4f}ms".format(1000 * np.median(times)))
