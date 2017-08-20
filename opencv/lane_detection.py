import argparse
import cv2
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description='Lane Detection')
parser.add_argument('-i','--input_image', help='Input Image',  default="lane.jpeg", type=str)
parser.add_argument('-roi','--roi', help='Use region of interest',  default=0, type=int)
parser.add_argument('-t','--threshold', help='Hough line threshold',  default=110, type=int)
parser.add_argument('-l','--low', help='Canny threshold',  default=100, type=int)
parser.add_argument('-high','--high', help='Canny threshold',  default=250, type=int)
parser.add_argument('-k','--kernel', help='Gaussian kernel',  default=7, type=int)

args = parser.parse_args()

filename = args.input_image
img = cv2.imread(filename)

low_threshold = args.low
high_threshold = args.high
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if args.kernel > 0:
    kernel_size = args.kernel
    gray = gaussian_blur(gray, kernel_size)

canny = cv2.Canny(gray, low_threshold, high_threshold)

n_rows, n_cols = gray.shape
if args.roi == 1:
    dst = roi_rectangle(canny, 350,n_rows,100,n_cols-100)
else:
    dst = canny
#rho and theta are the distance and angular resolution of the grid in Hough space
rho = 2
theta = np.pi/180.0
#threshold is minimum number of intersections in a grid for candidate line to go to output
threshold = args.threshold
min_line_len = 50
max_line_gap = 200

result = hough_lines(dst, rho, theta, threshold, min_line_len, max_line_gap)
result = weighted_img(result, img, alpha=0.8, beta=1., _lambda=0.)

# Polygon
# pts = np.array([[350,320],[n_cols-300,320],[n_cols-80,n_rows],[100, n_rows]], np.int32)
# # pts = pts.T
# pts = pts.reshape((-1,1,2))
# cv2.polylines(img,[pts],True,(0,255,255))
#
# # imshape = img.shape
# lower_left = [imshape[1]/9,imshape[0]]
# lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
# top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
# top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
# vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
# img = region_of_interest(img, vertices)

# img = region_of_interest(img, pts)
# dst = roi()
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]

# cv2.imshow('original',img)
cv2.imshow('gray', gray)

cv2.imshow('canny', canny)
if args.roi == 1:
    cv2.imshow('roi', dst)
cv2.imshow('result', result)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
