from __future__ import print_function, division

import cv2
import numpy as np


def iterateMinibatches(inputs, targets, batchsize, shuffle=False):
    """
    Iterator that creates minibatches
    :param inputs: (numpy tensor)
    :param targets: (numpy array)
    :param batchsize: (int)
    :param shuffle: (bool)
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def preprocessImage(image, width, height):
    """
    Preprocessing script to convert image into neural net input array
    :param image: (cv2 image)
    :param width: (int)
    :param height: (int)
    :return: (numpy array)
    """
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    x = image.flatten()
    # Normalize
    x = x / 255.
    x -= 0.5
    x *= 2
    return x


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
                b[i // 2] = f['arr_%d' % i]
            else:
                W[i // 2] = f['arr_%d' % i]

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
