from __future__ import print_function, with_statement, division

import os
import argparse

import cv2
import numpy as np


# Arrow keys
UP_KEY = 82
DOWN_KEY = 84
RIGHT_KEY = 83
LEFT_KEY = 81
ENTER_KEY = 10
SPACE_KEY = 32
EXIT_KEYS = [113, 27]  # Escape and q
S_KEY = 115  # Save key

output_folder = "dataset/video1"

parser = argparse.ArgumentParser(description='White Lane Detection for a batch of images')
parser.add_argument('-i','--input_video', help='Input Video',  default="debug/robot_vue.mp4", type=str)
args = parser.parse_args()

# OpenCV 3.x.x compatibility
if not hasattr(cv2, 'cv'):
    # 0-based index of the frame to be decoded/captured next.
    image_zero_index = cv2.CAP_PROP_POS_FRAMES
    frame_count = cv2.CAP_PROP_FRAME_COUNT
else:
    image_zero_index = cv2.cv.CV_CAP_PROP_POS_FRAMES
    frame_count = cv2.cv.CV_CAP_PROP_FRAME_COUNT

video_file = args.input_video
cap = cv2.VideoCapture(video_file)



current_idx = cap.get(image_zero_index)
n_frames = int(cap.get(frame_count))
print("{} frames".format(n_frames))
while True:
    while True:
        flag, img = cap.read()
        if flag:
            break
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(image_zero_index, current_idx - 1)
            cv2.waitKey(1000)
            continue

    original_img = img.copy()
    cv2.imshow("current", img)
    key = cv2.waitKey(0) & 0xff

    if key in EXIT_KEYS:
        cv2.destroyAllWindows()
        exit()
    elif key in [LEFT_KEY, RIGHT_KEY]:
        current_idx += 1 if key == RIGHT_KEY else -1
        current_idx = np.clip(current_idx, 0, n_frames-1)
        # Save image
        path = 'train/{}/{}.jpg'.format(output_folder, int(current_idx))
        cv2.imwrite(path, original_img)
        print("Saved {}".format(int(current_idx)))

    cap.set(image_zero_index, current_idx)
