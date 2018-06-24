from __future__ import print_function, division, absolute_import

import subprocess

import pytest
import torch as th

from .common import *
from constants import *
from train.train import main
from train.utils import adjustLearningRate, loadNetwork
from train.models import CustomNet


training_args = ['-f', DATASET, '--num_epochs', NUM_EPOCHS, '-bs', BATCHSIZE,
                 '--seed', SEED]
training_args = list(map(str, training_args))


def testTrain():
    for model_type in ['custom', 'cnn', 'mlp']:
        args = training_args + ['--model_type', model_type]
        ok = subprocess.call(['python', '-m', 'train.train'] + args)
        assertEq(ok, 0)
        weights = "{}_model_tmp.pth".format(model_type)
        loadNetwork(weights, model_type=model_type)


def testFailIfModelNotSupported():
    with pytest.raises(ValueError):
        main(DATASET, device="cuda", model_type="dummy_model")


def testTrainFromSavedModel():
    args = training_args + ['--model_type', MODEL_TYPE, '--load_model', WEIGHTS_PTH]
    ok = subprocess.call(['python', '-m', 'train.train'] + args)
    assertEq(ok, 0)


def testTestScript():
    args = ['-f', DATASET, '--model_type', MODEL_TYPE, '-w', WEIGHTS_PTH, '--no-display']
    ok = subprocess.call(['python', '-m', 'train.test'] + args)
    assertEq(ok, 0)


def testBenchmarkScript():
    args = ['-n', '1', '--model_type', MODEL_TYPE, '-w', WEIGHTS_PTH]
    ok = subprocess.call(['python', '-m', 'train.benchmark'] + args)
    assertEq(ok, 0)


def testAdjustLearningRate():
    model = CustomNet()
    optimizer = th.optim.SGD(model.parameters(), lr=1)
    adjustLearningRate(optimizer, epoch=1, n_epochs=2, lr_init=1, batch=10,
                           n_batch=10, method='cosine')
    current_lr = optimizer.param_groups[0]['lr']
    assertEq(current_lr, 0)

    decay_rate = 0.9
    adjustLearningRate(optimizer, epoch=5, n_epochs=10, lr_init=1, batch=1,
                           n_batch=10, method='multistep', decay_rate=decay_rate)
    current_lr = optimizer.param_groups[0]['lr']
    assertEq(current_lr, decay_rate)

    adjustLearningRate(optimizer, epoch=8, n_epochs=10, lr_init=1, batch=1,
                           n_batch=10, method='multistep', decay_rate=decay_rate)
    current_lr = optimizer.param_groups[0]['lr']
    assertEq(current_lr, decay_rate ** 2)
