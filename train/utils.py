from __future__ import print_function, division, absolute_import

import pickle as pkl

import cv2
import numpy as np
import torch as th
from sklearn.model_selection import train_test_split

from constants import INPUT_DIM, HEIGHT, WIDTH
from .models import MlpNetwork


def loadPytorchNetwork(model_name="mlp_model_tmp", n_hidden=None):
    """
    Load a saved pytorch model
    :param model_name: (str)
    :param n_hidden: ([int])
    :return: (pytorch model)
    """
    if '.pth' in model_name:
        model_name = model_name.split('.pth')[0]
    model = MlpNetwork(INPUT_DIM, n_hidden=n_hidden)
    model.load_state_dict(th.load(model_name + '.pth'))
    model.eval()
    return model


def saveToNpz(model, output_name="mlp_model_tmp"):
    """
    :param model: (PyTorch Model)
    :param output_name: (str)
    """
    np.savez(output_name, *[p.data.numpy().T for _, p in model.named_parameters()])


def loadDataset(split_seed=42, folder='', split=True, augmented=True):
    """
    Load the training images and preprocess them
    :param split_seed: (int) split_seed for pseudo-random generator
    :param folder: (str) input folder
    :param split: (bool) Whether to split the dataset into 3 subsets (train, validation, test)
    :param augmented: (bool) Whether to use data augmentation
    :return:
    """

    # Load the dataset info file (pickle object)
    with open('{}/infos.pkl'.format(folder), 'rb') as f:
        images_dict = pkl.load(f)['images']

    # Sort names
    images = list(images_dict.keys())
    images.sort()
    images_path = []

    # Load one image to retrieve original shape
    tmp_im = cv2.imread('{}/{}.jpg'.format(folder, images_dict[images[0]]['output_name']))
    height, width, _ = tmp_im.shape
    n_images = len(images)
    # If we use data augmentation we double the size of training data
    if augmented:
        images_path_augmented = []
        n_images *= 2

    X = np.zeros((n_images, INPUT_DIM), dtype=np.float32)
    y = np.zeros((n_images,), dtype=np.float32)

    print("original_shape=({},{})".format(width, height))
    print("resized_shape=({},{})".format(WIDTH, HEIGHT))

    for idx, name in enumerate(images):
        x_center, y_center = images_dict[name]['label']
        # Normalize output
        y[idx] = x_center / width

        image_path = '{}/{}.jpg'.format(folder, images_dict[name]['output_name'])
        im = cv2.imread(image_path)
        # Resize and normalize input
        X[idx, :] = preprocessImage(im, WIDTH, HEIGHT)
        images_path.append(path + '.jpg')
        # Flip the image+label to have more training data
        if augmented:
            horizontal_flip = cv2.flip(im, 1)
            X[len(images) + idx, :] = preprocessImage(horizontal_flip, WIDTH, HEIGHT)
            y[len(images) + idx] = (width - x_center) / width
            images_path_augmented.append(path + '.jpg')

    # By convention, augmented data are at the end
    if augmented:
        images_path += images_path_augmented

    print("Input tensor shape: ", X.shape)

    if not split:
        return X, y, images_path

    # For CNN reshape the data to 3D tensors
    # X = X.reshape((-1, WIDTH, HEIGHT, 3))
    # X = np.transpose(X, (0, 3, 2, 1))

    # Split the data into three subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=split_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=split_seed)

    return X_train, y_train, X_val, y_val, X_test, y_test


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
    Compute Mean Square Error
    and print its value for the different sets
    :param y_test: (numpy 1D array)
    :param y_true: (numpy 1D array)
    :param indices: [[int]] Indices of the different subsets
    """
    idx_train, idx_val, idx_test = indices
    # MSE Loss
    error = np.square(y_test - y_true)

    print('Train error={:.6f}'.format(np.mean(error[idx_train])))
    print('Val error={:.6f}'.format(np.mean(error[idx_val])))
    print('Test error={:.6f}'.format(np.mean(error[idx_test])))
    print('Total error={:.6f}'.format(np.mean(error)))
