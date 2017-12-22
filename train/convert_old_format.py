from __future__ import absolute_import, division

import os
import pickle as pkl
import argparse
from collections import defaultdict

import cv2

from constants import REGIONS

parser = argparse.ArgumentParser(description='Convert dataset to new format')
parser.add_argument('-i', '--input_folder', help='Input folder', default="", type=str, required=True)
parser.add_argument('-o', '--output_folder', help='Output folder', default="", type=str, required=True)
args = parser.parse_args()

images_path = [name for name in os.listdir(args.input_folder) if name.endswith('.jpg')]

images_path.sort()

infos_dict = {'input_folder': args.input_folder, 'images': defaultdict(dict)}
try:
    with open('{}/infos.pkl'.format(args.output_folder), 'rb') as f:
        infos_dict = pkl.load(f)
    print("infos.pkl will be updated")
except IOError:
    pass


should_exit = False
j = 0
for idx, name in enumerate(images_path):
    if 'hori' in name:
        continue
    im = cv2.imread('{}/{}'.format(args.input_folder, name))
    cx, cy = map(int, name.split('_')[0].split('-'))
    # Retrieve region idx
    i = int(name.split('-')[-1].split('.jpg')[0].split('r')[1])
    output_name = '{}'.format(j)
    idx = '{}_r{}'.format(name, i)
    infos_dict['images'][idx]['input_image'] = name
    infos_dict['images'][idx]['label'] = [cx, cy]
    infos_dict['images'][idx]['region'] = list(REGIONS[i])
    infos_dict['images'][idx]['output_name'] = output_name
    cv2.imwrite('{}/{}.jpg'.format(args.output_folder, output_name), im)
    j += 1


# Update infos
with open('{}/infos.pkl'.format(args.output_folder), 'wb') as f:
    pkl.dump(infos_dict, f)
