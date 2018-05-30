"""
Apply image processing on a video:
it processes each frame (line detection + line curve estimation)
and shows the result
"""
from __future__ import print_function, with_statement, division

import argparse

import cv2
import numpy as np

from constants import RIGHT_KEY, LEFT_KEY, SPACE_KEY, EXIT_KEYS
from image_processing.image_processing import processImage

# Pressing the space bar, it plays the video
playing_video = False

parser = argparse.ArgumentParser(description='Line Detection on a video')
parser.add_argument('-i', '--input_video', help='Input Video', default="video.mp4", type=str)
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

# Creating a window for later use
cv2.namedWindow('result')


def nothing(x):
    pass


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

    processImage(img, debug=True)

    if not playing_video:
        key = cv2.waitKey(0) & 0xff
    else:
        key = cv2.waitKey(10) & 0xff

    if key in EXIT_KEYS:
        cv2.destroyAllWindows()
        exit()
    elif key in [LEFT_KEY, RIGHT_KEY] or playing_video:
        current_idx += 1 if key == RIGHT_KEY or playing_video else -1
        current_idx = np.clip(current_idx, 0, n_frames - 1)
    elif key == SPACE_KEY:
        playing_video = not playing_video

    cap.set(image_zero_index, current_idx)
