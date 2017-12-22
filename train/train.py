#!/usr/bin/env python
"""
Train a neural network to detect a black&white line
"""
from __future__ import print_function, division

import argparse
import os
import time

import cv2
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import DenseLayer
from lasagne.regularization import regularize_network_params, l2
from lasagne.updates import adam
from sklearn.model_selection import train_test_split

seed = 42
np.random.seed(seed)
evaluate_print = 1  # Print info every 1 epoch
WIDTH, HEIGHT = 80, 20  # Shape of the resized input image fed to our model


def loadNetwork(model_name="mlp_model"):
    """
    Load a trained network and return
    prediction function along with the network object
    :param model_name: (str)
    :return: (lasagne network object, theano function)
    """
    input_var = T.matrix('inputs')
    input_dim = WIDTH * HEIGHT * 3
    network = buildMlp(input_var, input_dim)

    with np.load('{}.npz'.format(model_name)) as f:
        param_values = [f['arr_%d' % i].astype(np.float32) for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    pred_fn = theano.function([input_var], test_prediction)
    return network, pred_fn


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


def augmentDataset(in_folder='cropped', out_folder='augmented_dataset'):
    """
    Data augmentation (horizontal mirror)
    :param in_folder: (str)
    :param out_folder: (str)
    """
    images = [name for name in os.listdir(in_folder)]
    for idx, name in enumerate(images):
        r = name.split('.jpg')[0][-2:]  # Retrieve the ROI name
        cx, cy = map(int, name.split('_')[0].split('-'))
        image_path = '{}/{}'.format(in_folder, images[idx])
        image = cv2.imread(image_path)
        height, width, n_channels = image.shape
        horizontal_flip = cv2.flip(image, 1)
        cv2.imwrite('{}/{}-{}_{}-{}.jpg'.format(out_folder, cx, cy, idx, r), image)
        cv2.imwrite('{}/{}-{}_hori_{}-{}.jpg'.format(out_folder, width - cx, cy, idx, r), horizontal_flip)


def loadDataset(seed=42, folder='cropped', split=True):
    """
    Load the training images and preprocess them
    :param seed: (int) seed for pseudo-random generator
    :param folder: (str) input folder
    :param split: (bool) Whether to split the dataset into 3 subsets (train, validation, test)
    :return:
    """
    images = [name for name in os.listdir(folder)]

    tmp_im = cv2.imread('{}/{}'.format(folder, images[0]))
    height, width, n_channels = tmp_im.shape

    X = np.zeros((len(images), WIDTH * HEIGHT * n_channels), dtype=np.float64)
    y = np.zeros((len(images),), dtype=np.float64)

    print("original_shape=({},{})".format(width, height))
    print("resized_shape=({},{})".format(WIDTH, HEIGHT))
    factor = width / WIDTH

    for idx, name in enumerate(images):
        x_center, y_center = map(int, name.split('_')[0].split('-'))
        # TODO: check the formula below (if this changes, it must be changed in image_processing.py too)
        x_center /= factor * width
        y[idx] = x_center

        image_path = '{}/{}'.format(folder, images[idx])
        im = cv2.imread(image_path)
        X[idx, :] = preprocessImage(im, WIDTH, HEIGHT)

    print(X.shape)

    if not split:
        return X, y, images, factor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)

    return X_train, y_train, X_val, y_val, X_test, y_test


def buildMlp(input_var, input_dim):
    """
    Create the feedfoward neural net
    :param input_var: (Theano tensor)
    :param input_dim: (int)
    :return: (lasagne network object)
    """
    relu = lasagne.nonlinearities.rectify
    # linear = lasagne.nonlinearities.linear
    net = lasagne.layers.InputLayer(shape=(None, input_dim), input_var=input_var)
    net = lasagne.layers.DropoutLayer(net, p=0.1)
    net = DenseLayer(net, num_units=8, nonlinearity=relu)
    net = DenseLayer(net, num_units=4, nonlinearity=relu)
    l_out = DenseLayer(net, num_units=1, nonlinearity=relu)
    return l_out


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


def main(folder, num_epochs=1000, batchsize=1, learning_rate=0.0001, seed=42):
    """
    :param folder: (str)
    :param num_epochs: (int)
    :param batchsize: (int)
    :param learning_rate: (float)
    :param seed: (int)
    """
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = loadDataset(folder=folder, seed=seed)

    target_var = T.vector('targets')
    input_var = T.matrix('inputs')
    input_dim = X_train.shape[1]
    network = buildMlp(input_var, input_dim)
    model_name = "mlp_model"
    # Create prediction function
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    # Add L2 penalty
    loss += 1e-4 * regularize_network_params(network, l2)

    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = nesterov_momentum(loss, params, learning_rate=0.0001, momentum=0.8)
    updates = adam(loss, params, learning_rate=learning_rate)
    # Deterministic prediction function
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], test_loss)
    best_params, best_error = None, np.inf

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        # Full pass on training data
        # Update the model after each minibatch
        for batch in iterateMinibatches(X_train, y_train, batchsize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # Do a full pass on validation data
        val_err = 0
        val_batches = 0
        for batch in iterateMinibatches(X_val, y_val, batchsize, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1
        val_error = val_err / val_batches
        # Save the new best model
        if val_error < best_error:
            best_error = val_error
            best_params = lasagne.layers.get_all_param_values(network)
            np.savez('{}.npz'.format(model_name), *best_params)

        if (epoch + 1) % evaluate_print == 0:
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    # After training, we compute and print the test error:
    lasagne.layers.set_all_param_values(network, best_params)
    test_err = 0
    test_batches = 0
    for batch in iterateMinibatches(X_test, y_test, batchsize, shuffle=False):
        inputs, targets = batch
        err = val_fn(inputs, targets)
        test_err += err
        test_batches += 1
    print("Final results:")
    print("  best validation loss:\t\t{:.6f}".format(best_error))
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    # Save best model
    np.savez('{}.npz'.format(model_name), *best_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('--num_epochs', help='Number of epoch', default=3000, type=int)
    parser.add_argument('-bs', '--batchsize', help='Batch size', default=1, type=int)
    parser.add_argument('--seed', help='Random Seed', default=42, type=int)
    parser.add_argument('-f', '--folder', help='Training folder', default="augmented_dataset", type=str)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=1e-5, type=float)
    args = parser.parse_args()

    seed = args.seed
    main(folder=args.folder, num_epochs=args.num_epochs, batchsize=args.batchsize, learning_rate=args.learning_rate)
