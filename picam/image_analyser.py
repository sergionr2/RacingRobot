from __future__ import division, print_function

import threading
import time

try:
    import queue
except ImportError:
    import Queue as queue

import picamera.array
import cv2

from image_processing.image_processing import processImage
from constants import CAMERA_RESOLUTION, RECORD_VIDEO

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
        self.frame_queue = queue.Queue(maxsize=1)
        self.exit = False
        self.out_queue = out_queue
        self.debug = debug
        self.thread = None
        self.start()

    def analyse(self, frame):
        """
        Override base function
        :param frame: BGR image
        """
        self.frame_queue.put(item=frame, block=True)

    def extractInfo(self):
        # times = []
        try:
            while not self.exit:
                try:
                    frame = self.frame_queue.get(block=True, timeout=1)
                except queue.Empty:
                    print("Frame queue empty")
                    continue
                # 1 ms per loop
                # TODO: check that this conversion is not needed
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.debug:
                    self.out_queue.put(item=frame, block=False)
                else:
                    try:
                        # 10 ms per loop
                        # start_time = time.time()
                        turn_percent, centroids = processImage(frame)
                        # times.append(time.time() - start_time)
                        self.out_queue.put(item=(turn_percent, centroids), block=False)
                    except Exception as e:
                        print("Exception in RBGAnalyser processing image: {}".format(e))
                self.frame_num += 1
        except Exception as e:
            print("Exception in RBGAnalyser after loop: {}".format(e))
        # s_per_loop_image = np.mean(times)
        # print("Image processing: {:.2f}ms per loop | {} fps".format(s_per_loop_image * 1000, int(1 / s_per_loop_image)))

    def start(self):
        t = threading.Thread(target=self.extractInfo)
        self.thread = t
        t.deamon = True
        t.start()

    def stop(self):
        self.exit = True
        self.thread.join()
        self.frame_queue.queue.clear()


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
        # TODO: try with mode 6, larger FoV (works only with v2 module)
        self.camera.sensor_mode = 6
        self.camera.resolution = resolution
        print(self.camera.resolution)
        self.camera.framerate = fps
        self.out_queue = out_queue
        # self.camera.zoom = (0.0, 0.0, 1.0, 1.0)
        # self.camera.awb_gains = 1.5
        self.camera.awb_mode = 'auto'
        self.camera.exposure_mode = 'auto'
        self.debug = debug
        self.analyser = None

    def start(self):
        self.analyser = RGBAnalyser(self.camera, self.out_queue, debug=self.debug)
        self.camera.start_recording(self.analyser, format='bgr')
        if RECORD_VIDEO:
            self.camera.start_recording('debug/{}.h264'.format(experiment_time),
                                        splitter_port=2, resize=CAMERA_RESOLUTION)

    def stop(self):
        self.camera.wait_recording()
        if RECORD_VIDEO:
            self.camera.stop_recording(splitter_port=2)
        self.camera.stop_recording()
        self.analyser.stop()


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
