from __future__ import print_function, division

import time
import argparse

import cv2
import numpy as np
import torch as th
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from train.train import loadDataset, loadNetwork, loadPytorchNetwork
from constants import WIDTH, HEIGHT, RIGHT_KEY, LEFT_KEY, EXIT_KEYS

# Init seaborn
sns.set()

np.set_printoptions(precision=2)

def plot_representation(x, y, colors, name="", add_colorbar=True):
    fig = plt.figure(name)
    plt.scatter(x, y, s=7, c=colors, cmap='coolwarm', linewidths=0.1)
    plt.xlabel('State dimension 1')
    plt.ylabel('State dimension 2')
    plt.title(fill(name, TITLE_MAX_LENGTH))
    fig.tight_layout()
    if add_colorbar:
        plt.colorbar(label='x center')
    plt.show()


def plot_input_weights(weights, name='Input Weights', cmap='coolwarm'):
    fig = plt.figure(name)
    m, n = len(weights), 3
    axes = fig.subplots(m, n)
    v_min, v_max = np.min(weights), np.max(weights)
    labels = ['R', 'G', 'B']

    for i in range(len(weights)):
        for j in range(3):
            ax = axes[i, j]
            im = ax.imshow(weights[i, ..., j], vmin=v_min, vmax=v_max, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == len(weights) - 1:
                ax.set_xlabel(labels[j])
            if j == 0 :
                ax.set_ylabel("Unit {}".format(i))

    cax, kwargs = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
    plt.colorbar(im, cax=cax, **kwargs)
    plt.draw()
    plt.pause(0.0001)
    # plt.show()

def plot_matrix(matrix, name='Matrix', cmap='coolwarm'):
    fig = plt.figure(name)
    plt.clf()
    im = plt.imshow(matrix, cmap=cmap)
    im.axes.grid(False)
    plt.colorbar()
    plt.draw()
    plt.pause(0.0001)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for plotting weights and internal representations')
    parser.add_argument('-f', '--folder', help='Training folder', default="", type=str, required=True)
    parser.add_argument('-w', '--weights', help='Saved weights', default="", type=str, required=True)
    # parser.add_argument('--t-sne', action='store_true', default=False, help='Use t-SNE instead of PCA')
    parser.add_argument('--no-data-augmentation', action='store_true', default=False, help='Disables data augmentation')
    args = parser.parse_args()

    # Load dataset
    X, y_true, images = loadDataset(folder=args.folder, split=False, augmented=not args.no_data_augmentation)
    np.random.seed(1)
    permutation = np.random.permutation(len(X))
    X = X[permutation]
    y_true = y_true[permutation]
    images = np.array(images)[permutation]
    model = loadPytorchNetwork(args.weights)
    y_test, a1, a2 = model.customForward(Variable(th.from_numpy(X)))
    a1, a2, = a1.data.numpy(), a2.data.numpy()
    y_test = y_test.data.numpy()[:, 0]

    print("Max:{}".format(np.max(a1, axis=0)))
    print("Median:{}".format(np.median(a1, axis=0)))
    print("Mean:{}".format(np.mean(a1, axis=0)))
    print("Std:{}".format(np.std(a1, axis=0)))
    print("Null percentage:")
    print(100 * np.sum(a1 <= 0, axis=0) / len(a1))

    plt.ion()
    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx == 0:
            # Weights connected to input image
            plot_input_weights(param.data.numpy().reshape((-1, HEIGHT, WIDTH, 3)), name)
            continue
        w = param.data.numpy()
        if len(w.shape) == 1:
            w = w.reshape(-1, 1)
        plot_matrix(w, name)
    plt.ioff()
    plt.show()

    current_idx = 0
    while True:
        name = images[current_idx]
        im = cv2.imread('{}/{}'.format(args.folder, images[current_idx]))

        height, width, n_channels = im.shape

        if not args.no_data_augmentation and permutation[current_idx] >= len(images) // 2:
            im = cv2.flip(im, 1)

        x_center = int(y_test[current_idx] * width)
        x_center = np.clip(x_center, 0, width)
        y_center = height // 2
        cv2.circle(im, (x_center, y_center), radius=10, color=(0, 0, 255),
                   thickness=2, lineType=8, shift=0)

        cv2.imshow('Input Image', im)
        for name, a in zip(['Activations Layer 1', 'Activations Layer 2'], [a1, a2]):
            plot_matrix(a[current_idx].reshape((-1, 1)), name)

        key = cv2.waitKey(0) & 0xff
        if key in EXIT_KEYS:
            cv2.destroyAllWindows()
            break
        elif key in [LEFT_KEY, RIGHT_KEY]:
            current_idx += 1 if key == RIGHT_KEY else -1
            current_idx = np.clip(current_idx, 0, len(images) - 1)
