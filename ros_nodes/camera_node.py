from __future__ import division, print_function

import threading
import time

try:
    import queue
except ImportError:
    import Queue as queue

import picamera.array
import cv2

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage


class RGBAnalyser(picamera.array.PiRGBAnalysis):
    """
    Class used to retrieve an image from the picamera
    and process it
    :param camera: (PiCamera object)
    :param image_publisher: (ROS Publisher)
    """

    def __init__(self, camera, image_publisher):
        super(RGBAnalyser, self).__init__(camera)
        self.frame_num = 0
        self.referenceFrame = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.exit = False
        self.bridge = CvBridge()
        self.image_publisher = image_publisher
        self.start()

    def analyse(self, frame):
        """
        :param frame: BGR image object
        """
        self.frame_queue.put(item=frame, block=True)

    def extractInfo(self):
        try:
            while not self.exit:
                try:
                    frame = self.frame_queue.get(block=True, timeout=1)
                except queue.Empty:
                    print("Queue empty")
                    continue
                try:
                    # Publish new image
                    msg = self.bridge.cv2_to_imgmsg(frame, 'rgb8')
                    if not self.exit:
                        self.image_publisher.publish(msg)
                except CvBridgeError as e:
                    print("Error Converting cv image: {}".format(e.message))
                self.frame_num += 1
        except Exception as e:
            print("Exception after loop: {}".format(e))
            raise

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
    :param image_publisher: (ROS publisher)
    :param resolution: (int, int)
    :param fps: (int)
    """

    def __init__(self, image_publisher, resolution, fps=90):
        self.camera = picamera.PiCamera()
        # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        self.camera.sensor_mode = 7
        self.camera.resolution = resolution
        print(self.camera.resolution)
        self.camera.framerate = fps
        self.image_publisher = image_publisher
        # self.camera.zoom = (0.0, 0.0, 1.0, 1.0)
        # self.camera.awb_gains = 1.5
        self.camera.awb_mode = 'auto'
        self.exposure_mode = 'auto'

    def start(self):
        self.analyser = RGBAnalyser(self.camera, self.image_publisher)
        self.camera.start_recording(self.analyser, format='rgb')

    def stop(self):
        self.camera.wait_recording()
        self.camera.stop_recording()
        self.analyser.stop()

if __name__ == '__main__':
    image_publisher = rospy.Publisher('/picamera/image', Image, queue_size=10)
    resolution = (640 // 4, 480 // 4)
    fps = 30
    print("Starting PiCameraNode")
    rospy.init_node('PiCameraNode', anonymous=True)

    v = Viewer(image_publisher, resolution, fps)
    start_time = time.time()
    v.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    v.stop()

    print('FPS: {:.2f}'.format(v.analyser.frame_num / (time.time() - start_time)))
