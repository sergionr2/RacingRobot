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

from constants import NUM_OUTPUT
from .utils import JsonDataset, loadLabels
from .models import ConvolutionalNetwork, CustomNet

evaluate_print = 1  # Print info every 1 epoch
VAL_BATCH_SIZE = 64  # Batch size for validation and test data


def main(folder, num_epochs=100, batchsize=32,
         learning_rate=0.0001, seed=42, device="cpu", random_flip=0.5,
         model_type="cnn", evaluate_print=1, load_model=""):

    if not folder.endswith('/'):
        folder += '/'

    train_labels, val_labels, test_labels, _ = loadLabels(folder)

    # Seed the random generator
    np.random.seed(seed)
    th.manual_seed(seed)
    if device == "cuda":
        th.cuda.manual_seed(seed)

    # Retrieve number of samples per set
    n_train, n_val, n_test = len(train_labels), len(val_labels), len(test_labels)

    # Keywords for pytorch dataloader
    kwargs = {'num_workers': 1, 'pin_memory': False} if device == "cuda" else {}
    # Create data loaders
    train_loader = th.utils.data.DataLoader(
        JsonDataset(train_labels, preprocess=True, folder=folder, random_flip=random_flip),
        batch_size=batchsize, shuffle=True, **kwargs)

    # Random flip also for val ?
    val_loader = th.utils.data.DataLoader(JsonDataset(val_labels, preprocess=True, folder=folder),
                                          batch_size=VAL_BATCH_SIZE, shuffle=False, **kwargs)

    test_loader = th.utils.data.DataLoader(JsonDataset(test_labels, preprocess=True, folder=folder),
                                           batch_size=VAL_BATCH_SIZE, shuffle=False, **kwargs)

    model_name = "{}_model_tmp".format(model_type)
    if model_type == "cnn":
        model = ConvolutionalNetwork(num_output=NUM_OUTPUT, drop_p=0.0)
    elif model_type == "custom":
        model = CustomNet(num_output=NUM_OUTPUT)
    else:
        raise ValueError("Model type not supported")

    model = model.to(device)

    # L2 penalty
    # weight_decay = 1e-4
    weight_decay = 0
    # Optimizers
    # optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = th.optim.SGD(model.parameters(), lr=learning_rate,
                             momentum=0.9, weight_decay=weight_decay, nesterov=True)

    # Loss functions
    loss_fn = nn.MSELoss(size_average=False)
    # loss_fn = nn.SmoothL1Loss(size_average=False)
    best_error = np.inf
    best_model_path = "{}.pth".format(model_name)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
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
            # Move tensors to gpu if necessary
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        # Do a full pass on validation data
        model.eval()
        val_loss = 0
        with th.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # We don't need to compute gradient
                predictions = model(inputs)
                loss = loss_fn(predictions, targets)
                val_loss += loss.item()

        # Compute error per sample
        val_error = val_loss / n_val

        if val_error < best_error:
            best_error = val_error
            # Move back weights to cpu
            # Save Weights
            th.save(model.to("cpu").state_dict(), best_model_path)

            model.to(device)

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
    with th.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # We don't need to compute gradient
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            test_loss += loss.item()

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
    parser.add_argument('--model_type', help='Model type: cnn', default="cnn", type=str, choices=['cnn', 'custom'])
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=1e-3, type=float)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and th.cuda.is_available()
    device = th.device("cuda" if args.cuda else "cpu")
    main(folder=args.folder, num_epochs=args.num_epochs, batchsize=args.batchsize,
         learning_rate=args.learning_rate, device=device,
         seed=args.seed, load_model=args.load_model, model_type=args.model_type)
