from __future__ import print_function, division, absolute_import

import subprocess

NUM_EPOCHS = 2
DATASET = 'datasets/test_dataset'
SEED = 0
BATCHSIZE = 4

def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)

def assertNeq(left, right):
    assert left != right, "{} == {}".format(left, right)


def testTrain():
    for model_type in ['custom', 'cnn']:
        args = ['-f', DATASET, '--num_epochs', NUM_EPOCHS, '-bs', BATCHSIZE,
                '--seed', SEED, '--model_type', model_type]

        args = list(map(str, args))

        ok = subprocess.call(['python', '-m',  'train.train'] + args)
        assertEq(ok, 0)
