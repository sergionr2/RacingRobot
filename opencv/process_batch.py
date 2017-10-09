"""
Process a folder of images
"""
from __future__ import print_function, with_statement, division

import os
import argparse

import cv2
import numpy as np

from image_processing import processImage

# Arrow keys
UP_KEY = 82
DOWN_KEY = 84
RIGHT_KEY = 83
LEFT_KEY = 81
ENTER_KEY = 10
EXIT_KEYS = [113, 27]  # Escape and q
M_KEY = 109
L_KEY = 108
images = ['jpg', 'jpeg', 'png', 'gif']

parser = argparse.ArgumentParser(description='White Lane Detection for a batch of images')
parser.add_argument('-i', '--input_image', help='Input Image', default="", type=str)
parser.add_argument('-f', '--folder', help='Folder', default="", type=str)
parser.add_argument('-r', '--regions', help='ROI', default=1, type=int)
args = parser.parse_args()

if args.input_image != "" or args.folder != "":
    imgs = [args.input_image]
    if args.folder != "":
        imgs = [args.folder + '/' + im for im in os.listdir(args.folder) if im.split('.')[-1] in images]
    current_idx = 0
    while True:
        img = cv2.imread(imgs[current_idx])
        # r = [margin_left, margin_top, width, height]
        regions = None
        if args.regions == 0:
            regions = [[0, 0, img.shape[1], img.shape[0]]]
        processImage(img, debug=True, regions=regions)

        key = cv2.waitKey(0) & 0xff
        if key in EXIT_KEYS:
            cv2.destroyAllWindows()
            exit()
        elif key in [LEFT_KEY, RIGHT_KEY]:
            current_idx += 1 if key == RIGHT_KEY else -1
            current_idx = np.clip(current_idx, 0, len(imgs) - 1)
