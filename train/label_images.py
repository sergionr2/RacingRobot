"""
Tool for labeling images
"""
from __future__ import print_function, division

import os
import pickle as pkl
import argparse
from collections import defaultdict

import cv2

from opencv.image_processing import processImage
from constants import REGIONS

parser = argparse.ArgumentParser(description='Labeling tool for line detection')
parser.add_argument('-i', '--input_folder', help='Input Folder', default="", type=str, required=True)
parser.add_argument('-o', '--output_folder', help='Output folder', default="", type=str, required=True)

args = parser.parse_args()


input_folder = args.input_folder
output_folder = args.output_folder
images = [im for im in os.listdir(input_folder) if im.endswith('.jpg')]

infos_dict = {'input_folder': input_folder, 'images': defaultdict(dict)}
try:
    with open('{}/infos.pkl'.format(output_folder), 'rb') as f:
        infos_dict = pkl.load(f)
    print("infos.pkl will be updated")
except IOError:
    pass


idx_images = [(int(im.split('.jpg')[0]), im) for im in images]
# idx_images = [(int(im.split('.jpg')[0].split('_')[1]), im) for im in images]
# Sort the images
idx_images = sorted(idx_images, key=lambda x: x[0])
images = [im for _, im in idx_images]

print(len(images))

j = 0
should_exit = False
for name in images:
    if should_exit:
        break
    img = cv2.imread('{}/{}'.format(input_folder, name))
    print(name)

    original_img = img.copy()
    for i, r in enumerate(REGIONS):
        # img will be modified by processImage()
        img = original_img.copy()
        margin_left, margin_top, _, _ = r
        im_cropped = original_img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        centroids, errors, exit_loop = processImage(img, debug=True, regions=[r], interactive=True)
        if not all(errors):
            # Save the labeled image (and store the label in the name)
            x, y = centroids.flatten()

            cx, cy = x - margin_left, y - margin_top
            output_name = '{}'.format(j)
            idx = '{}_r{}'.format(name, i)
            infos_dict['images'][idx]['input_image'] = name
            infos_dict['images'][idx]['label'] = [cx, cy]
            infos_dict['images'][idx]['region'] = list(r)
            infos_dict['images'][idx]['output_name'] = output_name
            cv2.imwrite('{}/{}.jpg'.format(output_folder, output_name), im_cropped)
            # Update infos
            with open('{}/infos.pkl'.format(output_folder), 'wb') as f:
                # protocol=2 for python 2 compatibility
                pkl.dump(infos_dict, f, protocol=2)
        j += 1

        if exit_loop:
            should_exit = True
            break
