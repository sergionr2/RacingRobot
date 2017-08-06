from __future__ import division, print_function

import time
import threading
try:
    import queue
except ImportError:
    import Queue as queue

# NOTE: with multiprocessing picamera does not seem to work
# import multiprocessing

import picamera.array
import cv2
import numpy as np

from opencv.moments import processImage

class ImageProcessingThread(threading.Thread):
    def __init__(self, viewer, condition):
        super(ImageProcessingThread, self).__init__()
        self.deamon = True
        self.v = viewer
        self.condition = condition

    def run(self):
        v = self.v
        start_time = time.time()
        v.start()

        with self.condition:
            self.condition.wait()
        v.stop()

        print('FPS: {:.2f}'.format(v.analyser.frame_num / (time.time() - start_time)))


class RGBAnalyser(picamera.array.PiRGBAnalysis):
    def __init__(self, camera, out_queue):
        super(RGBAnalyser, self).__init__(camera)
        self.frame_num = 0
        self.referenceFrame = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop = False
        self.out_queue = out_queue
        self.data = 0
        self.start()

    def analyse(self, frame):
       self.frame_queue.put(item=frame, block=True)

    def extractInfo(self):
        try:
            while not self.stop:
                frame = self.frame_queue.get(block=True, timeout=2)
                cx, cy, error = processImage(frame)
                # print(cx, cy)
                self.out_queue.put(item=(cx, cy, error), block=False)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # self.out_queue.put(item=frame, block=False)
                self.frame_num += 1
        except:
            pass

    def start(self):
        t = threading.Thread(target=self.extractInfo)
        self.thread = t
        t.deamon = True
        t.start()

    def stop(self):
        self.frame_queue.queue.clear()
        self.stop = True



class Viewer(object):
    def __init__(self, out_queue, resolution):
        self.camera = picamera.PiCamera()
        # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        self.camera.sensor_mode = 7
        self.camera.resolution = resolution
        print(self.camera.resolution)
        self.camera.framerate = 90
        self.out_queue = out_queue
        # self.camera.zoom = (0.0, 0.0, 1.0, 1.0)
        # self.camera.awb_gains = 1.5
        self.camera.awb_mode = 'auto'
        self.exposure_mode = 'auto'

    def start(self):
        self.analyser = RGBAnalyser(self.camera, self.out_queue)
        self.camera.start_recording(self.analyser, format='rgb')

    def stop(self):
        self.camera.wait_recording()
        self.camera.stop_recording()


if __name__ == '__main__':
    out_queue = queue.Queue()
    condition_lock = threading.Lock()
    condition = threading.Condition(condition_lock)
    resolution = (640, 480)
    image_thread = ImageProcessingThread(Viewer(out_queue, resolution), condition)
    image_thread.start()
    time.sleep(1)
    # End the thread
    with condition:
        condition.notify_all()
    image_thread.join()
    for i in range(10):
        out_queue.get()
    cv2.imwrite("test.jpg", out_queue.get())
