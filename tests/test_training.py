from __future__ import print_function, division, absolute_import

import subprocess

import pytest

from .common import *
from constants import *
from train.train import main

training_args = ['-f', DATASET, '--num_epochs', NUM_EPOCHS, '-bs', BATCHSIZE,
                 '--seed', SEED]
training_args = list(map(str, training_args))


def testTrain():
    for model_type in ['custom', 'cnn', 'mlp']:
        args = training_args + ['--model_type', model_type]
        ok = subprocess.call(['python', '-m', 'train.train'] + args)
        assertEq(ok, 0)


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
