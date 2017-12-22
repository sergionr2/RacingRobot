"""
Tool for labeling images
"""
import os

import cv2

from opencv.image_processing import processImage
from constants import EXIT_KEYS


id_video = "23sept"
input_folder = 'opencv/train/dataset/video_sun/debug/'
output_folder = 'opencv/train/cropped/video_sun/'
images = [im for im in os.listdir(input_folder) if im.endswith('.jpg')]

# idx_images = [(int(im.split('.jpg')[0]), im) for im in images]
idx_images = [(int(im.split('.jpg')[0].split('_')[1]), im) for im in images]
# Sort the images
idx_images = sorted(idx_images, key=lambda x: x[0])
images = [im for _, im in idx_images]

print(len(images))

for idx, name in enumerate(images):
    img = cv2.imread('{}/{}'.format(input_folder, name))
    print(name)
    original_idx = name.split(".jpg")[0]
    original_img = img.copy()
    img = original_img.copy()
    max_width = img.shape[1]
    # Regions of interest
    r0 = [0, 150, max_width, 50]
    r1 = [0, 125, max_width, 50]
    r2 = [0, 100, max_width, 50]
    r3 = [0, 75, max_width, 50]
    r4 = [0, 50, max_width, 50]
    regions = [r1, r2, r3]  # Keep only the one we are using during test

    original_img = img.copy()
    for i, r in enumerate(regions):
        img = original_img.copy()
        margin_left, margin_top, _, _ = r
        im_cropped = original_img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        centroids, errors = processImage(img, debug=True, regions=[r], interactive=True)
        if not all(errors):
            # Save the labeled image (and store the label in the name)
            x, y = centroids.flatten()
            cx, cy = x - margin_left, y - margin_top
            cv2.imwrite('{}/{}-{}_{}-{}{}r{}.jpg'.format(output_folder, cx, cy, idx,
                                                         id_video, original_idx, i),
                        im_cropped)

            # key = cv2.waitKey(0) & 0xff
            # if key in EXIT_KEYS:
            #     break
