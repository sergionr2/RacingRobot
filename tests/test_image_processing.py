from __future__ import print_function, division, absolute_import

import subprocess

from .common import *
from constants import *
from image_processing.image_processing import processImage
from image_processing.warp_image import transformPoints, warpImage

np.random.seed(0)
test_image = np.random.randint(0, 255, size=(MAX_WIDTH, MAX_HEIGHT, 3), dtype=np.uint8)

def testProcessImage():
    processImage(test_image)
    processImage(test_image, debug=True)


def testWarpDemo():
    args = ['-i', TEST_IMAGE_PATH, '--no-display']
    ok = subprocess.call(['python', '-m', 'image_processing.warp_image'] + args)
    assertEq(ok, 0)


def testWarping():
    x, y = processImage(test_image, debug=True)
    transformPoints(x, y)
    warpImage(test_image)
