from __future__ import print_function, with_statement, division

import argparse

import cv2
import numpy as np

from constants import REF_ANGLE, MAX_ANGLE
from opencv.train.train import preprocessImage, loadNetwork, WIDTH, HEIGHT


def loadVanillaNet(weights_npy='mlp_model.npz'):
    """
    Load a trained network and
    return the forward function in pure numpy
    :param weights_npy: (str) path to the numpy archive
    :return: (function) the neural net forward function
    """
    W, b = {}, {}
    with np.load(weights_npy) as f:
        print("Loading network")
        n_layers = len(f.files) // 2
        for i in range(len(f.files)):
            if i % 2 == 1:
                b[i // 2] = f['arr_%d' % i]
            else:
                W[i // 2] = f['arr_%d' % i]

    def relu(x):
        """
        Rectify activation function: f(x) = max(0, x)
        :param x: (numpy array)
        :return: (numpy array)
        """
        y = x.copy()
        y[y < 0] = 0
        return y

    def forward(X):
        """
        Forward pass of a fully-connected neural net
        with rectifier activation function
        :param X: (numpy tensor)
        :return: (numpy array)
        """
        a = X
        for i in range(n_layers):
            z = np.dot(a, W[i]) + b[i]
            a = relu(z)
        return a

    return forward


# Either load network with theano or with numpy
# network, pred_fn = loadNetwork()
pred_fn = loadVanillaNet()


def mouseCallback(event, x, y, flags, centers):
    """
    Callback in interactive (annotation) mode
    """
    # Save the position of the mouse on left click
    if event == cv2.EVENT_LBUTTONDOWN:
        centers[0] = (x, y)


def processImage(image, debug=False, regions=None, interactive=False):
    """
    :param image: (rgb image)
    :param debug: (bool)
    :param regions: [[int]]
    :param interactive: (bool)
    :return:(float, numpy array)
    """
    # r = [margin_left, margin_top, width, height]
    im_width = image.shape[1]
    if regions is None:
        # Regions of interest
        # r = [margin_left, margin_top, width, height]
        r0 = [0, 150, im_width, 50]
        r1 = [0, 125, im_width, 50]
        r2 = [0, 100, im_width, 50]
        r3 = [0, 75, im_width, 50]
        r4 = [0, 50, im_width, 50]
        regions = np.array([r1, r2, r3]) # Before [r0, r1, r2]

    centroids = np.zeros((len(regions), 2), dtype=int)
    errors = [False for _ in regions]

    if not debug:
        # Batch Prediction
        pred_imgs = []
        factor = im_width / WIDTH
        for idx, r in enumerate(regions):
            im_cropped = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            pred_imgs.append(preprocessImage(im_cropped, WIDTH, HEIGHT))
        centroids[:, 0] = pred_fn(np.array(pred_imgs))[:, 0] * factor * im_width
        centroids[:, 0] += regions[:, 0]
        centroids[:, 1] = regions[:, 3] // 2 + regions[:, 1]
    else:
        for idx, r in enumerate(regions):
            center = {}
            margin_left, margin_top, _, _ = r
            im_cropped = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

            im_cropped_tmp = im_cropped.copy()
            im_width = im_cropped_tmp.shape[1]
            factor = im_width / WIDTH
            pred_img = preprocessImage(im_cropped, WIDTH, HEIGHT)
            # Predict where is the center of the line using the trained network
            # WARNING: The scaling factor must be the same as the one used during training
            x_center = int(pred_fn([pred_img])[0] * factor * im_width)
            y_center = im_cropped_tmp.shape[0] // 2

            if debug:
                # Draw prediction and true center
                cv2.circle(im_cropped_tmp, (x_center, y_center), radius=10,
                           color=(0, 0, 255), thickness=2, lineType=8, shift=0)
                cv2.imshow('crop_pred{}'.format(idx), im_cropped_tmp)

                # display line: y = height // 2
                h = im_cropped.shape[0] // 2
                cv2.line(im_cropped, (0, h), (1000, h), (0, 0, 255), 2)
                cv2.imshow('crop{}'.format(idx), im_cropped)
                # Labeling mode
                if interactive:
                    cv2.setMouseCallback('crop{}'.format(idx), mouseCallback, center)
                    key = cv2.waitKey(0) & 0xff

            if debug and interactive:
                if center.get(0):
                    cx, cy = center[0]
                else:
                    cx, cy = 0, 0
                    errors[idx] = True
            else:
                cx, cy = x_center, y_center

            centroids[idx] = np.array([cx + margin_left, cy + margin_top])
    # Linear Regression to fit a line
    x = centroids[:, 0]
    y = centroids[:, 1]
    # Case x = cst
    if len(np.unique(x)) == 1:
        pts = centroids[:2, :]
        turn_percent = 0
    else:
        A = np.vstack([x, np.ones(len(x))]).T
        # y = m*x + b
        m, b = np.linalg.lstsq(A, y)[0]
        if debug:
            x = np.array([0, im_width], dtype=int)
            pts = (np.vstack([x, m * x + b]).T).astype(int)
        track_angle = np.arctan(m)
        diff_angle = abs(REF_ANGLE) - abs(track_angle)
        # Estimation of the line curvature
        turn_percent = (diff_angle / MAX_ANGLE) * 100

    if debug and len(centroids) > 1:
        a, b = pts
    else:
        pts = None
        a, b = (0, 0), (0, 0)

    if debug:
        if all(errors):
            # print("No centroids found")
            cv2.imshow('result', image)
        else:
            for cx, cy in centroids:
                cv2.circle(image, (cx, cy), radius=10, color=(0, 0, 255), thickness=1, lineType=8, shift=0)
                cv2.line(image, tuple(a), tuple(b), color=(100, 100, 0), thickness=2, lineType=8)
            cv2.line(image, (im_width // 2, 0), (im_width // 2, image.shape[0]),
                     color=(100, 0, 0), thickness=2, lineType=8)
            cv2.imshow('result', image)

    if interactive:
        return centroids, errors
    return turn_percent, centroids


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='White Lane Detection')
    parser.add_argument('-i', '--input_image', help='Input Image', default="", type=str)

    args = parser.parse_args()
    if args.input_image != "":
        img = cv2.imread(args.input_image)
        turn_percent, centroids = processImage(img, debug=True)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
            exit()
