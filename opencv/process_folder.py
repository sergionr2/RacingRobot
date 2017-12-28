"""
Apply image processing on a folder of images
"""
from __future__ import print_function, with_statement, division

import argparse

import cv2
import numpy as np
import os

from constants import RIGHT_KEY, LEFT_KEY, EXIT_KEYS
from opencv.image_processing import processImage

images = ['jpg', 'jpeg', 'png', 'gif']

parser = argparse.ArgumentParser(description='Line Detection for a folder of images')
parser.add_argument('-i', '--input_image', help='Input Image', default="", type=str)
parser.add_argument('-f', '--folder', help='Folder', default="", type=str)
parser.add_argument('-r', '--regions', help='whether to use regions of interests of the whole image', default=1, type=int)
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

        # Retrieve pressed key
        key = cv2.waitKey(0) & 0xff

        if key in EXIT_KEYS:
            cv2.destroyAllWindows()
            exit()
        elif key in [LEFT_KEY, RIGHT_KEY]:
            current_idx += 1 if key == RIGHT_KEY else -1
            current_idx = np.clip(current_idx, 0, len(imgs) - 1)
