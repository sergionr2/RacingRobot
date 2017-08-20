from __future__ import print_function, with_statement, division

import argparse

import cv2
import numpy as np

def processImage(image, debug=False):
    """
    :param image: (rgb image)
    :param debug: (bool)
    :return: (int, int)
    """
    error = False
    # r = [margin_left, margin_top, width, height]
    r = [50, 150, 200, 50]
    #r = [0,0, image.shape[0], image.shape[1]]
    margin_left, margin_top, _, _ = r

    imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    image = imCrop
    if debug:
        cv2.imshow('crop', imCrop)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # define range of blue color in HSV
    lower_white = np.array([0, 0, 212])
    upper_white = np.array([131, 255, 255])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([10, 10, 50])

    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel_erode = np.ones((4,4), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)

    kernel_dilate = np.ones((6,6),np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)

    if debug:
        cv2.imshow('mask', mask)
        cv2.imshow('eroded', eroded_mask)
        cv2.imshow('dilated', dilated_mask)

    # cv2.RETR_CCOMP  instead of cv2.RETR_TREE
    im2, contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    if debug:
        # Draw biggest
        # cv2.drawContours(image, contours, 0, (0,255,0), 3)
        cv2.drawContours(image, contours, -1, (0,255,0), 3)

    if len(contours) > 0:
        M = cv2.moments(contours[0])
        # Centroid
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    else:
        cx, cy = 0, 0
        error = True

    if debug:
        if error:
            print("No centroid found")
        else:
            print("Found centroid at ({}, {})".format(cx, cy))
        cv2.circle(image, (cx,cy), radius=10, color=(0,0,255),
                   thickness=1, lineType=8, shift=0)
        cv2.imshow('result', image)
    orignal_cx, original_cy = cx + margin_left, cy + margin_top
    return orignal_cx, original_cy, error

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='White Lane Detection')
    parser.add_argument('-i','--input_image', help='Input Image',  default="0.png", type=str)

    args = parser.parse_args()

    img = cv2.imread(args.input_image)
    cx, cy, error = processImage(img, debug=True)

    if not error:
        cv2.circle(img, (cx,cy), radius=10, color=(0,0,255),
                   thickness=1, lineType=8, shift=0)
        cv2.imshow('result', img)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
