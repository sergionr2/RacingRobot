from __future__ import print_function, division, absolute_import

import subprocess

from .common import *


def testTrain():
    for model_type in ['custom', 'cnn']:
        args = ['-f', DATASET, '--num_epochs', NUM_EPOCHS, '-bs', BATCHSIZE,
                '--seed', SEED, '--model_type', model_type]

        args = list(map(str, args))

        ok = subprocess.call(['python', '-m',  'train.train'] + args)
        assertEq(ok, 0)
