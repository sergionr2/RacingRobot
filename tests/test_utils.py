from __future__ import print_function, division, absolute_import

import numpy as np

from .common import *
from train.utils import preprocessImage, loadNetwork, predict
from constants import *

test_image = 234 * np.ones((*CAMERA_RESOLUTION, 3), dtype=np.uint8)

def testPreprocessing():
    image = preprocessImage(test_image, INPUT_WIDTH, INPUT_HEIGHT)
    # Check normalization
    assertEq(len(np.where(np.abs(image) > 1)[0]), 0)
    # Check resize
    assertEq(image.size, 3 * INPUT_WIDTH * INPUT_HEIGHT)
    # Check normalization on one element
    assertEq(image[0, 0, 0], ((234 / 255.) - 0.5) *  2)

def testPredict():
    model = loadNetwork(WEIGHTS_PTH, NUM_OUTPUT, MODEL_TYPE)
    x, y = predict(model, test_image)
