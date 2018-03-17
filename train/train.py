"""
Train a neural network to detect a black&white line
"""
from __future__ import print_function, division, absolute_import

import argparse
import time

import numpy as np
import torch as th
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable

from constants import SPLIT_SEED
from .utils import preprocessImage, saveToNpz, loadDataset, adjustLearningRate
from .models import MlpNetwork, ConvolutionalNetwork

evaluate_print = 1  # Print info every 1 epoch
VAL_BATCH_SIZE = 256  # Batch size for validation and test data


def main(folder, num_epochs=1000, batchsize=1,
         learning_rate=0.0001, seed=42, cuda=False,
         load_model=""):
    """
    :param folder: (str)
    :param num_epochs: (int)
    :param batchsize: (int)
    :param learning_rate: (float)
    :param seed: (int)
    :param cuda: (bool)
    :param load_model: (str) path to a saved model
    """
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = loadDataset(folder=folder, split_seed=SPLIT_SEED)

    # Seed the random generator
    np.random.seed(seed)
    th.manual_seed(seed)
    if cuda:
        th.cuda.manual_seed(seed)


    # Convert to torch tensor
    X_train, y_train = th.from_numpy(X_train), th.from_numpy(y_train).view(-1, 1)
    X_val, y_val = th.from_numpy(X_val), th.from_numpy(y_val).view(-1, 1)
    X_test, y_test = th.from_numpy(X_test), th.from_numpy(y_test).view(-1, 1)
    # Retrieve number of samples per set
    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)

    # Keywords for pytorch dataloader
    kwargs = {'num_workers': 1, 'pin_memory': False} if cuda else {}
    # Create data loaders
    train_loader = th.utils.data.DataLoader(th.utils.data.TensorDataset(X_train, y_train),
                                            batch_size=batchsize, shuffle=True, **kwargs)

    val_loader = th.utils.data.DataLoader(th.utils.data.TensorDataset(X_val, y_val),
                                          batch_size=VAL_BATCH_SIZE, shuffle=False, **kwargs)

    test_loader = th.utils.data.DataLoader(th.utils.data.TensorDataset(X_test, y_test),
                                           batch_size=VAL_BATCH_SIZE, shuffle=False, **kwargs)

    model = MlpNetwork(X_train.shape[1], n_hidden=[20, 4], drop_p=0.5)
    model_name = "mlp_model_tmp"
    # model_name = "cnn__model_tmp"
    # model = ConvolutionalNetwork()
    if load_model != "":
        model.load_state_dict(th.load(load_model))

    if cuda:
        model.cuda()

    # L2 penalty
    # weight_decay = 1e-4
    weight_decay = 0
    # Optimizers
    # optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = th.optim.SGD(model.parameters(), lr=learning_rate,
                             momentum=0.9, weight_decay=weight_decay, nesterov=True)
    # Learning rate schedulers
    # scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    # scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    # Loss functions
    loss_fn = nn.MSELoss(size_average=False)
    # loss_fn = nn.SmoothL1Loss(size_average=False)
    best_error = np.inf
    best_model_path = "{}.pth".format(model_name)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # scheduler.step()
        # Switch to training mode
        model.train()
        train_loss, val_loss = 0, 0
        start_time = time.time()
        # Full pass on training data
        # Update the model after each minibatch
        for i, (inputs, targets) in enumerate(train_loader):
            # Adjust learning rate
            # adjustLearningRate(optimizer, epoch, num_epochs, lr_init=learning_rate,
            #                         batch=i, n_batch=len(train_loader), method='multistep')
            # Move variables to gpu
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # Convert to pytorch variables
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
            # Set volatile to True because we don't need to compute gradient
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            val_loss += loss.data[0]

        # Compute error per sample
        val_error = val_loss / n_val
        # Save the new best model
        if val_error < best_error:
            best_error = val_error
            # Move back weights to cpu
            if cuda:
                model.cpu()
            # Save as pytorch (pth) and numpy file (npz)
            th.save(model.state_dict(), best_model_path)
            saveToNpz(model, model_name)

            if cuda:
                model.cuda()

        if (epoch + 1) % evaluate_print == 0:
            # Then we print the results for this epoch:
            # Losses are averaged over the samples
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
    parser.add_argument('-f', '--folder', help='Training folder', type=str, required=True)
    parser.add_argument('--num_epochs', help='Number of epoch', default=50, type=int)
    parser.add_argument('-bs', '--batchsize', help='Batch size', default=4, type=int)
    parser.add_argument('--seed', help='Random Seed', default=42, type=int)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--load_model', help='Start from a saved model', default="", type=str)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=1e-3, type=float)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and th.cuda.is_available()
    main(folder=args.folder, num_epochs=args.num_epochs, batchsize=args.batchsize,
         learning_rate=args.learning_rate, cuda=args.cuda,
         seed=args.seed, load_model=args.load_model)
