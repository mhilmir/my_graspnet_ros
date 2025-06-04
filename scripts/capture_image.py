#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import scipy.io as scio
import os
import numpy as np
from ultralytics import YOLO

class MyRealsense:
    def __init__(self):
        rospy.init_node('capture_image_node')
        self.bridge = CvBridge()

        self.color_msg = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_msg = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.camera_info_msg = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.cam_info_callback)

        self.color_image = None
        self.depth_image = None
        self.camera_matrix = None
        self.clicked_point = None
        
        cv2.namedWindow("Image Saver Masked Window")
        cv2.setMouseCallback("Image Saver Masked Window", self.click_event)

        self.bbox = (0, 0, 0, 0)  # Initialize the bounding box
        self.rate = rospy.Rate(30)  # 30 hz

        self.model = YOLO('yolov8n.pt')
        self.detected_boxes = []
    
    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            print("clicked", self.clicked_point)


    def color_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("Color CV Bridge Error: %s", e)
    
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
        except CvBridgeError as e:
            rospy.logerr("depth CV Bridge Error: %s", e)
    
    def cam_info_callback(self, msg):
        self.camera_matrix = msg

    def run(self):
        while not rospy.is_shutdown():

            color_image_copy = self.color_image.copy()

            # results = model(frame_rgb)
            results = self.model(color_image_copy)
            detected_boxes = []

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    # if box.conf[0] > 0.5:  # nambahin ini
                    x1, y1, x2, y2 = map(int, box)
                    detected_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(color_image_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if self.clicked_point:
                cx, cy = self.clicked_point
                for x1, y1, x2, y2 in detected_boxes:
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        self.bbox = (x1, y1, x2 - x1, y2 - y1)
                        break
                self.clicked_point = None

            if self.bbox != (0, 0, 0, 0):
                x, y, w, h = self.bbox
                # mask the image by bbox
                masked_depth = np.zeros_like(self.depth_image)
                masked_depth[y:y+h, x:x+w] = self.depth_image[y:y+h, x:x+w]

                cv2.imshow("masked_depth_img", masked_depth)
            
                fx = self.camera_matrix.K[0]
                fy = self.camera_matrix.K[4]
                cx = self.camera_matrix.K[2]
                cy = self.camera_matrix.K[5]

                # Create output directory if not exists
                output_dir = rospy.get_param('~output_dir', '/tmp/ros_images')
                os.makedirs(output_dir, exist_ok=True)

                # Save images
                color_path = os.path.join(output_dir, 'color_ros.png')
                depth_path = os.path.join(output_dir, 'depth_ros.png')

                cv2.imwrite(color_path, self.color_image)

                # Graspnet needs depth in meters
                if masked_depth.dtype == np.uint16:
                    # Depth in millimeters (RealSense/ZED/others)
                    depth_np = masked_depth.astype(np.float32) / 1000.0 
                elif masked_depth.dtype == np.float32:
                    # Depth in meters
                    depth_np = masked_depth.copy()
                else:
                    raise ValueError("Unsupported depth image type")

                np.save(os.path.join(output_dir, 'depth_ros.npy'), depth_np)  # Save as numpy array to preserve float values
                cv2.imwrite(depth_path, (depth_np * 1000).astype(np.uint16))  # Save visualization as PNG

                meta = {
                    'intrinsic_matrix': np.array([
                        [fx, 0,  cx],
                        [0,  fy, cy],
                        [0,  0,  1]
                    ], dtype=np.float32),
                    'factor_depth': 1.0  # Storing in millimeters
                }
                meta_path = os.path.join(output_dir, 'meta.mat')
                scio.savemat(meta_path, meta)

                rospy.loginfo(f"Color image saved to: {color_path}")
                rospy.loginfo(f"Depth image saved to: {depth_path}")
                rospy.loginfo(f"CameraInfo saved to: {meta_path}")

                break


            cv2.imshow("Image Saver Masked Window", color_image_copy)
            cv2.imshow("depth_img", self.depth_image)
            
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            self.rate.sleep()
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    my_realsense = MyRealsense()
    my_realsense.run()