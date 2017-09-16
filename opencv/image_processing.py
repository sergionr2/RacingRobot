from __future__ import print_function, with_statement, division

import argparse

import cv2
import numpy as np

from train.train import preprocessImage, loadNetwork, WIDTH, HEIGHT, WIDTH_CNN


REF_ANGLE = - np.pi / 2
use_network = True
if use_network:
    cnn = False
    network, pred_fn = loadNetwork(cnn=cnn)

def processImage(image, debug=False, regions=None, thresholds=None):
    """
    :param image: (rgb image)
    :param debug: (bool)
    :param regions: [[int]]
    :param thresholds: (dict)
    :return: (int, int)
    """
    error = False
    # r = [margin_left, margin_top, width, height]
    max_width = image.shape[1]
    if regions is None:
        r0 = [0, 150, max_width, 50]
        r1 = [0, 125, max_width, 25]
        r2 = [0, 100, max_width, 25]
        # r1 = [0, 125, max_width, 50]
        # r2 = [0, 100, max_width, 50]
        regions = [r0, r1]
    centroids = np.zeros((len(regions), 2), dtype=int)
    errors = [False for _ in regions]
    for idx, r in enumerate(regions):
        margin_left, margin_top, _, _ = r
        im_cropped = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        if use_network:
            im_cropped_tmp = im_cropped.copy()
            im_width = im_cropped_tmp.shape[1]
            if cnn:
                global WIDTH
                WIDTH = WIDTH_CNN
            factor = im_width / WIDTH
            pred_img = preprocessImage(im_cropped, WIDTH, HEIGHT, cnn=cnn)
            x_center = int(pred_fn([pred_img])[0] * factor * im_width)
            y_center = im_cropped_tmp.shape[0] // 2
            # Draw prediction and true center
            cv2.circle(im_cropped_tmp, (x_center, y_center), radius=10, color=(0,0,255),
            thickness=2, lineType=8, shift=0)
            cv2.imshow('crop_pred{}'.format(idx), im_cropped_tmp)

        if debug:
            cv2.imshow('crop{}'.format(idx), im_cropped)

        hsv = cv2.cvtColor(im_cropped, cv2.COLOR_RGB2HSV)
        # define range of blue color in HSV
        if thresholds is not None:
            lower_white = thresholds['lower_white']
            upper_white = thresholds['upper_white']
        else:
            lower_white = np.array([0, 0, 0])
            #upper_white = np.array([131, 255, 255])
            upper_white = np.array([85, 255, 255])

            # lower_black = np.array([0, 0, 0])
            # upper_black = np.array([16, 16, 26])

        # Threshold the HSV image
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # mask = cv2.inRange(hsv, lower_black, upper_black)

        kernel_erode = np.ones((4,4), np.uint8)
        eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)

        kernel_dilate = np.ones((4,4),np.uint8)
        dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)

        if debug:
            cv2.imshow('mask{}'.format(idx), mask)
            cv2.imshow('eroded{}'.format(idx), eroded_mask)
            cv2.imshow('dilated{}'.format(idx), dilated_mask)

        contour_result = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # OpenCV 2.x
        if cv2.__version__.split('.')[0] == '2':
            contours, hierarchy = contour_result
        else:
            # cv2.RETR_CCOMP  instead of cv2.RETR_TREE
            im2, contours, hierarchy = contour_result

        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        if debug and not use_network:
            # Draw biggest
            # cv2.drawContours(im_cropped, contours, 0, (0,255,0), 3)
            cv2.drawContours(im_cropped, contours, -1, (0,255,0), 3)

        if len(contours) > 0:
            M = cv2.moments(contours[0])
            # Centroid
            try:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            except ZeroDivisionError:
                cx, cy = 0, 0
                errors[idx] = True
        else:
            cx, cy = 0, 0
            errors[idx] = True

        if use_network:
            cx, cy = x_center, y_center

        centroids[idx] = np.array([cx + margin_left, cy + margin_top])
    if False:
        pass
    # Linear Regression to fit a line
    x = centroids[:,0]
    y = centroids[:, 1]
    # Case x = cst
    if len(np.unique(x)) == 1:
        pts = centroids[:2,:]
        turn_percent = 0
    else:
        # FIXME: take only centroids with no error
        x = np.array([x[0], x[-1]])
        y = np.array([y[0], y[-1]])
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y)[0]
        # y = m*x + b
        x = np.array([0, image.shape[1]], dtype=int)
        pts = np.vstack([x, m * x + b]).T
        pts = pts.astype(int)
        track_angle = np.arctan(m)
        diff_angle = abs(REF_ANGLE) - abs(track_angle)
        max_angle = 2 * np.pi / 3
        turn_percent = (diff_angle / max_angle) * 100
    if len(centroids) > 2:
        a,b = pts
    else:
        pts = None
        a,b = (0,0), (0,0)

    if debug:
        if all(errors):
            print("No centroids found")
            cv2.imshow('result', image)
        else:
            for cx, cy in centroids:
                cv2.circle(image, (cx,cy), radius=10, color=(0,0,255),
                           thickness=1, lineType=8, shift=0)

            cv2.line(image, tuple(a), tuple(b), color=(100,100,0),
                     thickness=2, lineType=8)
            cv2.line(image, (image.shape[1]//2, 0), (image.shape[1]//2, image.shape[0]), color=(100,0,0),thickness=2, lineType=8)
            cv2.imshow('result', image)
    return pts, turn_percent, centroids, errors

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='White Lane Detection')
    parser.add_argument('-i','--input_image', help='Input Image',  default="", type=str)

    args = parser.parse_args()
    if args.input_image != "":
        img = cv2.imread(args.input_image)
        pts, turn_percent, centroids, errors = processImage(img, debug=True)
        # turn_mat = np.array([turn_percent, any(errors)]).reshape((1, -1))
        # error = (img.shape[0]//2 - centroids[-1,0]) / (img.shape[0]//2)
        # error_mat = np.array([error, turn_percent]).reshape((1, -1))
        # mat = np.vstack((pts, turn_mat, error_mat))
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
            exit()
