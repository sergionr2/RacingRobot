import os

import cv2
import numpy as np

from image_processing import processImage

EXIT_KEYS = [113, 27]  # Escape and q

input_folder = 'before_crop'
output_folder = 'cropped'
images = [im for im in os.listdir(input_folder) if im.endswith('.jpg')]


for idx, im in enumerate(images):
    img = cv2.imread('{}/{}'.format(input_folder, im))
    # Get thresholds
    _min, _max = im.split('_max_')
    _max = _max.split(']')[0]

    # Remove Extra spaces
    _min = _min.replace('  ', ' ')
    _max = _max.replace('  ', ' ')
    try:
        h_min, s_min, v_min = map(int, _min[5:-1].split(' '))
    except ValueError:
        _min = [elem for elem in _min[6:-1].split(' ') if elem != '']
        h_min, s_min, v_min = map(int, _min)
    h_max, s_max, v_max = map(int, _max[2:].split(' '))
    thresholds = {
        'lower_white': np.array([h_min, s_min, v_min]),
        'upper_white': np.array([h_max, s_max, v_max])
    }
    # print(im)
    # print(thresholds)
    r0 = [0, 150, img.shape[1], 50]
    r = r0
    margin_left, margin_top, _, _ = r
    original_img = img.copy()
    imCrop = original_img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    pts, turn_percent, centroids, errors = processImage(img, debug=True, regions=[r0], thresholds=thresholds)
    if not all(errors):
        x, y = centroids.flatten()
        cx, cy = x - margin_left, y - margin_top
        cv2.imwrite('{}/{}-{}_{}.jpg'.format(output_folder, cx,cy,idx), imCrop)
    key = cv2.waitKey(0) & 0xff
    if key in EXIT_KEYS:
        break
