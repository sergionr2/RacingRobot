from __future__ import print_function, division

import cv2
import numpy as np
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# From https://github.com/pytorch/pytorch/commit/e9ef20eab5e5cf361bdc7a425c7f8b873baad9d3
class CosineAnnealingLR(_LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + np.cos(self.last_epoch / self.T_max * np.pi)) / 2
                for base_lr in self.base_lrs]


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


def computeMSE(y_test, y_true, indices):
    """
    :param y_test: (numpy 1D array)
    :param y_true: (numpy 1D array)
    :parma indices: [[int]]
    """
    idx_train, idx_val, idx_test = indices
    # MSE Loss
    error = np.square(y_test - y_true)

    print('Train error={:.6f}'.format(np.mean(error[idx_train])))
    print('Val error={:.6f}'.format(np.mean(error[idx_val])))
    print('Test error={:.6f}'.format(np.mean(error[idx_test])))
    print('Total error={:.6f}'.format(np.mean(error)))
