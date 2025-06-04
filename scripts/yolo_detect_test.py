#!/usr/bin/env python3

import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
### Always import torch and ultralytics before any ROS-related imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class YoloDetector:
    def __init__(self):
        rospy.init_node('yolo_detect_test_node', anonymous=True)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.detection_frame = None
        self.frame_count = 0
        self.detection_interval = 3
        self.confidence_threshold = 0.4

        # Load YOLO model
        rospy.loginfo("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')

        # Use CUDA if available
        if torch.cuda.is_available():
            rospy.loginfo("CUDA is available, using GPU")
            self.model.to('cuda')
        else:
            rospy.logwarn("CUDA not available, running on CPU (slow!)")

        # Subscribe to image topic
        self.image_sub = rospy.Subscriber(
            '/camera/color/image_raw', Image, self.image_callback, queue_size=1)

        rospy.loginfo("YOLOv8 detector initialized")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")

    def run_detection(self):
        if self.latest_frame is None:
            return None

        # Resize frame (smaller input = faster inference)
        frame = cv2.resize(self.latest_frame, (416, 416))

        start_time = time.time()

        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        fps = 1.0 / (time.time() - start_time)

        # Draw results
        result_frame = results[0].plot()

        # Add FPS counter
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return result_frame

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.latest_frame is not None:
                if self.frame_count % self.detection_interval == 0:
                    self.detection_frame = self.run_detection()

                if self.detection_frame is not None:
                    cv2.imshow("YOLOv8 Detection", self.detection_frame)
                else:
                    cv2.imshow("Camera Feed", self.latest_frame)

                key = cv2.waitKey(1)
                if key == 27:
                    break

                self.frame_count += 1

            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = YoloDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
