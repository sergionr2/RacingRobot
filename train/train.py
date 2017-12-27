#!/usr/bin/env python
"""
Train a neural network to detect a black&white line
"""
from __future__ import print_function, division, absolute_import

import argparse
import os
import time
import pickle as pkl

import lasagne
import theano
import theano.tensor as T
from lasagne.layers import DenseLayer

import cv2
import numpy as np
import torch as th
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from constants import WIDTH, HEIGHT, INPUT_DIM, SPLIT_SEED
from .utils import preprocessImage, CosineAnnealingLR
from .models import MlpNetwork, ConvolutionalNetwork

evaluate_print = 1  # Print info every 1 epoch
VAL_BATCH_SIZE = 256

# TODO: change std of weights initialization

def loadNetwork(model_name="mlp_model"):
    """
    Load a trained network and return
    prediction function along with the network object
    :param model_name: (str)
    :return: (lasagne network object, theano function)
    """
    # Remove npz
    if '.npz' in model_name:
        model_name = model_name.split('.npz')[0]
    input_var = T.matrix('inputs')
    network = buildMlp(input_var, INPUT_DIM)

    with np.load('{}.npz'.format(model_name)) as f:
        param_values = [f['arr_%d' % i].astype(np.float32) for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    pred_fn = theano.function([input_var], test_prediction)
    return network, pred_fn


def loadPytorchNetwork(model_name="mlp_model_tmp"):
    if '.pth' in model_name:
        model_name = model_name.split('.pth')[0]
    model = MlpNetwork(INPUT_DIM)
    model.load_state_dict(th.load(model_name + '.pth'))
    model.eval()
    return model


def saveToNpz(model, output_name="mlp_model_tmp"):
    """
    :param model: (PyTorch Model)
    """
    np.savez(output_name, *[p.data.numpy().T for _, p in model.named_parameters()])


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


def loadDataset(seed=42, folder='cropped', split=True, augmented=True):
    """
    Load the training images and preprocess them
    :param seed: (int) seed for pseudo-random generator
    :param folder: (str) input folder
    :param split: (bool) Whether to split the dataset into 3 subsets (train, validation, test)
    :param augmented: (bool)
    :return:
    """

    with open('{}/infos.pkl'.format(folder), 'rb') as f:
        images_dict = pkl.load(f)['images']

    images = list(images_dict.keys())
    images.sort()
    images_path = []

    # TODO: check channel order (BGR or RGB)
    tmp_im = cv2.imread('{}/{}.jpg'.format(folder, images_dict[images[0]]['output_name']))
    height, width, _ = tmp_im.shape
    n_images = len(images)
    if augmented:
        images_path_augmented = []
        n_images *= 2

    X = np.zeros((n_images, INPUT_DIM), dtype=np.float32)
    y = np.zeros((n_images,), dtype=np.float32)

    print("original_shape=({},{})".format(width, height))
    print("resized_shape=({},{})".format(WIDTH, HEIGHT))
    # factor = width / WIDTH

    for idx, name in enumerate(images):
        x_center, y_center = images_dict[name]['label']
        # TODO: check the formula below (if this changes, it must be changed in image_processing.py too)
        y[idx] = x_center / width

        path = images_dict[name]['output_name']
        image_path = '{}/{}.jpg'.format(folder, path)
        im = cv2.imread(image_path)
        X[idx, :] = preprocessImage(im, WIDTH, HEIGHT)
        images_path.append(path + '.jpg')
        if augmented:
            horizontal_flip = cv2.flip(im, 1)
            X[len(images) + idx, :] = preprocessImage(horizontal_flip, WIDTH, HEIGHT)
            images_path_augmented.append(path + '.jpg')
            y[len(images) + idx] = (width - x_center) / width

    if augmented:
        images_path += images_path_augmented

    print("Input tensor shape: ", X.shape)

    if not split:
        return X, y, images_path

    # for CNN
    # X = X.reshape((-1, WIDTH, HEIGHT, 3))
    # X = np.transpose(X, (0, 3, 2, 1))
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)

    return X_train, y_train, X_val, y_val, X_test, y_test


def main(folder, num_epochs=1000, batchsize=1, learning_rate=0.0001, seed=42, cuda=False):
    """
    :param folder: (str)
    :param num_epochs: (int)
    :param batchsize: (int)
    :param learning_rate: (float)
    :param seed: (int)
    :param cuda: (bool)
    """
    # Load the dataset
    print("Loading data...")

    X_train, y_train, X_val, y_val, X_test, y_test = loadDataset(folder=folder, seed=SPLIT_SEED)
    # Seed the random generator
    np.random.seed(seed)
    th.manual_seed(seed)
    if cuda:
        th.cuda.manual_seed(seed)


    kwargs = {'num_workers': 1, 'pin_memory': False} if cuda else {}

    # Convert to torch tensor
    X_train, y_train = th.from_numpy(X_train), th.from_numpy(y_train).view(-1, 1)
    X_val, y_val = th.from_numpy(X_val), th.from_numpy(y_val).view(-1, 1)
    X_test, y_test = th.from_numpy(X_test), th.from_numpy(y_test).view(-1, 1)
    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)

    # Create data loaders
    train_loader = th.utils.data.DataLoader(th.utils.data.TensorDataset(X_train, y_train),
                                            batch_size=batchsize, shuffle=True, **kwargs)

    val_loader = th.utils.data.DataLoader(th.utils.data.TensorDataset(X_val, y_val),
                                          batch_size=VAL_BATCH_SIZE, shuffle=False, **kwargs)

    test_loader = th.utils.data.DataLoader(th.utils.data.TensorDataset(X_test, y_test),
                                          batch_size=VAL_BATCH_SIZE, shuffle=False, **kwargs)

    input_dim = X_train.shape[1]
    model = MlpNetwork(input_dim, n_hidden=[20, 4], drop_p=0.6)
    model_name = "mlp_model_tmp"
    # model_name = "cnn__model_tmp"
    # model = ConvolutionalNetwork()

    if cuda:
        model.cuda()

    # L2 penalty
    # weight_decay = 1e-4
    weight_decay = 0
    # optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = th.optim.SGD(model.parameters(), lr=learning_rate,
                             momentum=0.9, weight_decay=weight_decay, nesterov=True)
    # scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
    # scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    # scheduler =  CosineAnnealingLR(optimizer, T_max=10, eta_min=0.05)

    loss_fn = nn.MSELoss(size_average=False)
    # loss_fn = nn.SmoothL1Loss(size_average=False)
    best_error = np.inf
    best_model_path = "{}.pth".format(model_name)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # scheduler.step()
        model.train()
        train_loss, val_loss = 0, 0
        start_time = time.time()
        # Full pass on training data
        # Update the model after each minibatch
        for inputs, targets in train_loader:
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()

        # Do a full pass on validation data
        model.eval()
        val_loss = 0
        for inputs, targets in val_loader:
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            val_loss += loss.data[0]

        val_error = val_loss / n_val
        # Save the new best model
        if val_error < best_error:
            best_error = val_error
            if cuda:
                model.cpu()

            th.save(model.state_dict(), best_model_path)
            saveToNpz(model, model_name)

            if cuda:
                model.cuda()

        if (epoch + 1) % evaluate_print == 0:
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_loss / n_train))
            print("  validation loss:\t\t{:.6f}".format(val_error))

    # After training, we compute and print the test error:
    model.load_state_dict(th.load(best_model_path))
    test_loss = 0
    for inputs, targets in test_loader:
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        test_loss += loss.data[0]
    print("Final results:")
    print("  best validation loss:\t\t{:.6f}".format(best_error))
    print("  test loss:\t\t\t{:.6f}".format(test_loss / n_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('--num_epochs', help='Number of epoch', default=50, type=int)
    parser.add_argument('-bs', '--batchsize', help='Batch size', default=4, type=int)
    parser.add_argument('--seed', help='Random Seed', default=42, type=int)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    parser.add_argument('-f', '--folder', help='Training folder', default="augmented_dataset", type=str)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=1e-4, type=float)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and th.cuda.is_available()
    main(folder=args.folder, num_epochs=args.num_epochs, batchsize=args.batchsize,
         learning_rate=args.learning_rate, cuda=args.cuda, seed=args.seed)
