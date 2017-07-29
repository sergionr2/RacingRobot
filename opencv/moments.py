import argparse

import cv2
import numpy as np

parser = argparse.ArgumentParser(description='White Lane Detection')
parser.add_argument('-i','--input_image', help='Input Image',  default="0.png", type=str)

args = parser.parse_args()

img = cv2.imread(args.input_image)

r = [50, 150, 400, 100]
imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
cv2.imshow('crop', imCrop)

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# define range of blue color in HSV
lower_white = np.array([0, 0, 215])
# lower_white = np.array([0, 0, 200])
upper_white = np.array([10, 10, 255])

# lower_black = np.array([0, 0, 0])
# upper_black = np.array([10, 10, 50])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_white, upper_white)
# mask = cv2.inRange(hsv, lower_black, upper_black)

kernel_erode = np.ones((4,4), np.uint8)
eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)

kernel_dilate = np.ones((6,6),np.uint8)
dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)

cv2.imshow('mask', mask)
cv2.imshow('eroded', eroded_mask)
cv2.imshow('dilated', dilated_mask)

# cv2.RETR_CCOMP  instead of cv2.RETR_TREE
im2, contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Sort by area
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
# Draw biggest
# cv2.drawContours(img, contours, 0, (0,255,0), 3)
cv2.drawContours(img, contours, -1, (0,255,0), 3)

M = cv2.moments(contours[0])
# Centroid
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

cv2.circle(img, (cx,cy), radius=10, color=(0,0,255), thickness=1, lineType=8, shift=0)

cv2.imshow('result', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
