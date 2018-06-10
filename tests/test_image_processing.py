from __future__ import print_function, division, absolute_import

import subprocess

import pytest

from .common import *
from constants import *
from image_processing.image_processing import processImage

np.random.seed(0)
test_image = np.random.randint(0, 255, size=(MAX_WIDTH, MAX_HEIGHT, 3), dtype=np.uint8)

def testProcessImage():
    processImage(test_image)
    processImage(test_image, debug=True)
