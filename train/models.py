"""
Different neural network architectures for detecting the line
# TODO: change std of weights initialization
"""
from __future__ import print_function, division, absolute_import

import torch.nn as nn
import torch.nn.functional as F


class MlpNetwork(nn.Module):
    """
    Dense Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W) (to be consistent with CNN network)
    :param input_dim: (int) 3 x H x H
    :param n_hidden: [int]
    :param drop_p: (float) Dropout proba
    """

    def __init__(self, input_dim, n_hidden=None, drop_p=0.0):
        super(MlpNetwork, self).__init__()
        if n_hidden is None:
            n_layer1 = 20
            n_layer2 = 4
        else:
            n_layer1, n_layer2 = n_hidden
        self.fc1 = nn.Linear(input_dim, n_layer1)
        self.fc2 = nn.Linear(n_layer1, n_layer2)
        self.fc3 = nn.Linear(n_layer2, 1)
        self.drop_p = drop_p
        self.activation_fn = F.relu

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.activation_fn(self.fc3(x))
        return x

    def customForward(self, x):
        """
        Return intermediate results
        """
        x = x.view(x.size(0), -1)
        x1 = self.activation_fn(self.fc1(x))
        x2 = self.activation_fn(self.fc2(x1))
        x = self.activation_fn(self.fc3(x2))
        return x, x1, x2


class ConvolutionalNetwork(nn.Module):
    """
    Convolutional Neural Network
    input shape : 3-channel RGB images of shape (3 x H x W)
    """

    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv_layers = nn.Sequential(

            # 20x80x3 -> 9x39x64
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Linear(9 * 39 * 8, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
