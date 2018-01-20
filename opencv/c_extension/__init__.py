from __future__ import print_function, absolute_import, division

import numpy as np
from train.utils import loadWeights
from .fast_image_processing import processImage as fastProcessImage
from .fast_image_processing import setWeights, forward

# Load weights from npz file
W, b = loadWeights()
# Load the weights in the c++ extension
setWeights(W[0], b[0], W[1], b[1], W[2], b[2])
