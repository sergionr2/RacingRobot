import argparse
import socket

import cv2
import numpy as np
# from scipy import stats

from opencv.noise import *

REF_ANGLE = - np.pi / 2

def processImage(image, debug=False):
    """
    :param image: (rgb image)
    :param debug: (bool)
    :return: (int, int)
    """
    error = False
    # r = [margin_left, margin_top, width, height]
    r0 = [50, 150, 450, 50]
    r1 = [50, 100, 450, 50]
    r2 = [50, 50, 450, 50]
    regions = [r0, r1, r2]
    centroids = np.zeros((len(regions), 2), dtype=int)
    errors = [False for _ in regions]
    for idx, r in enumerate(regions):
        margin_left, margin_top, _, _ = r
        imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        if debug:
            cv2.imshow('crop', imCrop)

        hsv = cv2.cvtColor(imCrop, cv2.COLOR_RGB2HSV)
        # define range of blue color in HSV
        lower_white = np.array([0,0,210])
        upper_white = np.array([23,255,255])

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([16, 16, 26])

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
            # cv2.drawContours(imCrop, contours, 0, (0,255,0), 3)
            cv2.drawContours(imCrop, contours, -1, (0,255,0), 3)

        if len(contours) > 0:
            M = cv2.moments(contours[0])
            # Centroid
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx, cy = 0, 0
            errors[idx] = True

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
        x = np.array([x[0], x[2]])
        y = np.array([y[0], y[2]])
        # slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        # m, b = slope, intercept
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
    a,b = pts
    # if track_angle > 0:
    #     print("LEFT {}".format(turn_percent))
    # else:
    #     print("RIGHT {}".format(turn_percent))

    if debug:
        if all(errors):
            print("No centroids found")
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
        pts, turn_percent, _, _ = processImage(img, debug=True)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
            exit()
    else:
        i = 0
        HOST = 'localhost'
        PORT = 50011
        should_stop = False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            while not should_stop:
                s.listen(1)
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024)
                    if not data: break
                    input_image = data.decode("utf-8")
                    if input_image == "stop":
                        should_stop = True
                        break
                    img = cv2.imread("blender/" + input_image)
                    noisedImage = rotateImage( img, random.random()*0.02-0.01, random.random()*0.02-0.01, random.random()*0.02-0.01 ) # 5 degrees
                    cv2.imwrite("blender/" + input_image, noisedImage)

                    pts, turn_percent, centroids, errors = processImage(noisedImage)
                    path = "render/{}".format(i)
                    turn_mat = np.array([turn_percent, any(errors)]).reshape((1, -1))
                    error = (img.shape[0]//2 - centroids[-1,0]) / (img.shape[0]//2)
                    error_mat = np.array([error, turn_percent]).reshape((1, -1))
                    mat = np.vstack((pts, turn_mat, error_mat))
                    np.save("blender/"+path, mat)
                    conn.sendall(bytes(path+".npy", 'utf-8'))
                    i += 1
