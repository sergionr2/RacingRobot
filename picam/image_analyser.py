from __future__ import division, print_function

import time
import threading
try:
    import queue
except ImportError:
    import Queue as queue

import picamera.array
import cv2
import numpy as np

from moments import processImage


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
                self.out_queue.put(item=(cx, cy, error), block=False)
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
    def __init__(self, out_queue):
        self.camera = picamera.PiCamera()
        # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        self.camera.sensor_mode = 7
        self.camera.resolution = (640//2, 480//2)
        print(self.camera.resolution)
        self.camera.framerate = 90
        self.out_queue = out_queue
        # self.camera.zoom = (0.0, 0.0, 1.0, 1.0)
        # self.camera.awb_gains = 1.5
        # self.camera.awb_mode = 'auto'
        # self.exposure_mode = 'auto'

    def start(self):
        self.analyser = RGBAnalyser(self.camera, self.out_queue)
        self.camera.start_recording(self.analyser, format='rgb')

    def stop(self):
        self.camera.wait_recording()
        self.camera.stop_recording()


if __name__ == '__main__':
    out_queue = queue.Queue()
    v = Viewer(out_queue)
    v.start()
    t = time.time()
    time.sleep(5)
    v.stop()
    print('FPS: {:.2f}'.format(v.analyser.frame_num / (time.time() - t)))
