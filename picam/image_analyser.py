from __future__ import division, print_function

import threading
import time

try:
    import queue
except ImportError:
    import Queue as queue

import picamera.array
import cv2

from opencv.image_processing import processImage
from constants import SAVE_EVERY

emptyException = queue.Empty
fullException = queue.Full

experiment_time = int(time.time())


class ImageProcessingThread(threading.Thread):
    """
    Thread used to retrieve image and do the image processing
    :param viewer: (Viewer object)
    :param exit_condition: (Condition object)
    """

    def __init__(self, viewer, exit_condition):
        super(ImageProcessingThread, self).__init__()
        self.deamon = True
        self.v = viewer
        self.exit_condition = exit_condition

    def run(self):
        v = self.v
        start_time = time.time()
        v.start()

        # Wait until the thread is notified to exit
        with self.exit_condition:
            self.exit_condition.wait()
        v.stop()

        print('FPS: {:.2f}'.format(v.analyser.frame_num / (time.time() - start_time)))


class RGBAnalyser(picamera.array.PiRGBAnalysis):
    """
    Class used to retrieve an image from the picamera
    and process it
    :param camera: (PiCamera object)
    :param out_queue: (Queue) queue used for output of image processing
    :param debug: (bool) set to true, queue will be filled with raw images
    """

    def __init__(self, camera, out_queue, debug=False):
        super(RGBAnalyser, self).__init__(camera)
        self.frame_num = 0
        self.referenceFrame = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop = False
        self.out_queue = out_queue
        self.data = 0
        self.debug = debug
        self.start()

    def analyse(self, frame):
        self.frame_queue.put(item=frame, block=True)

    def extractInfo(self):
        try:
            while not self.stop:
                frame = self.frame_queue.get(block=True, timeout=2)
                if self.debug:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.out_queue.put(item=frame, block=False)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.frame_num % SAVE_EVERY == 0:
                        cv2.imwrite("debug/{}_{}.jpg".format(experiment_time, self.frame_num), frame)
                        pass
                    try:
                        turn_percent, centroids = processImage(frame)
                        self.out_queue.put(item=(turn_percent, centroids), block=False)
                    except Exception as e:
                        print("Exception in image RBGAnalyser: {}".format(e))
                self.frame_num += 1
        except Exception as e:
            print("Exception in image RBGAnalyser after loop: {}".format(e))

    def start(self):
        t = threading.Thread(target=self.extractInfo)
        self.thread = t
        t.deamon = True
        t.start()

    def stop(self):
        self.frame_queue.queue.clear()
        self.stop = True


class Viewer(object):
    """
    Class that initialize the camera and start the PiCamera Thread
    :param out_queue: (Queue)
    :param resolution: (int, int)
    :param debug: (bool)
    :param fps: (int)
    """

    def __init__(self, out_queue, resolution, debug=False, fps=90):
        self.camera = picamera.PiCamera()
        # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        self.camera.sensor_mode = 7
        self.camera.resolution = resolution
        print(self.camera.resolution)
        self.camera.framerate = fps
        self.out_queue = out_queue
        # self.camera.zoom = (0.0, 0.0, 1.0, 1.0)
        # self.camera.awb_gains = 1.5
        self.camera.awb_mode = 'auto'
        self.exposure_mode = 'auto'
        self.debug = debug

    def start(self):
        self.analyser = RGBAnalyser(self.camera, self.out_queue, debug=self.debug)
        self.camera.start_recording(self.analyser, format='rgb')

    def stop(self):
        self.camera.wait_recording()
        self.camera.stop_recording()


if __name__ == '__main__':
    out_queue = queue.Queue()
    condition_lock = threading.Lock()
    exit_condition = threading.Condition(condition_lock)
    resolution = (640 // 2, 480 // 2)
    image_thread = ImageProcessingThread(Viewer(out_queue, resolution, debug=True), exit_condition)
    image_thread.start()
    time.sleep(5)
    # End the thread
    with exit_condition:
        exit_condition.notify_all()
    image_thread.join()
    i = 0
    while not out_queue.empty():
        print("picam/build/{}.jpg".format(i))
        cv2.imwrite("picam/build/{}.jpg".format(i), out_queue.get())
        i += 1
