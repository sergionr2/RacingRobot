from __future__ import print_function, division, absolute_import

NUM_EPOCHS = 2
DATASET = 'datasets/test_dataset'
SEED = 0
BATCHSIZE = 4

def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)

def assertNeq(left, right):
    assert left != right, "{} == {}".format(left, right)
