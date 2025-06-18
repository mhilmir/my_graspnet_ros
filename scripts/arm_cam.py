#!/usr/bin/env python3

import cv2
import scipy.io as scio
import os
import numpy as np
from ultralytics import YOLO
import math
### Always import torch and ultralytics before any ROS-related imports
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Point, Quaternion, Pose
from my_graspnet_ros.srv import GraspDetection

class MyRealsense:
    def __init__(self):
        rospy.init_node('arm_cam_node')
        self.bridge = CvBridge()

        self.image_pub = rospy.Publisher('/camera/arm/gripper_cam', Image, queue_size=1)
        self.yolo_enabled_pub = rospy.Publisher('/yolo_enabled_arm', Bool, queue_size=1)
        self.grasp_score_pub = rospy.Publisher('/grasp_result/score', Float32, queue_size=1)
        self.grasp_width_pub = rospy.Publisher('/grasp_result/width', Float32, queue_size=1)
        self.grasp_height_pub = rospy.Publisher('/grasp_result/height', Float32, queue_size=1)
        self.grasp_depth_pub = rospy.Publisher('/grasp_result/depth', Float32, queue_size=1)
        self.grasp_pose_pub = rospy.Publisher('/grasp_result/pose', Pose, queue_size=1)
        # self.bbox_real_pub = rospy.Publisher('/bbox_real', Point, queue_size=1)

        self.color_msg = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_msg = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.camera_info_msg = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.cam_info_callback)
        self.yolo_enabled_sub = rospy.Subscriber("/yolo_enabled_arm", Bool, self.yolo_enabled_callback)
        self.mouse_sub = rospy.Subscriber("/camera/arm/mouse_click", Point, self.mouse_callback)

        self.color_image = None
        self.depth_image = None
        self.camera_matrix = None
        self.clicked_point = None
        self.yolo_enabled = False  # Flag to control YOLO inference
        
        # cv2.namedWindow("Image Saver Masked Window")
        # cv2.setMouseCallback("Image Saver Masked Window", self.click_event)

        self.bbox = (0, 0, 0, 0)  # Initialize the bounding box
        self.rate = rospy.Rate(30)  # 30 hz

        self.model = YOLO('yolov8n.pt')
        self.detected_boxes = []
    
    # def click_event(self, event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         self.clicked_point = (x, y)
    #         print("clicked", self.clicked_point)

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

    def yolo_enabled_callback(self, msg):
        self.yolo_enabled = msg.data
        rospy.loginfo(f"YOLO enabled: {self.yolo_enabled}")
    
    def mouse_callback(self, msg):
        self.clicked_point = msg
        print("clicked\n", self.clicked_point)

    def pixel_to_real_world_coordinates(x_pixel, y_pixel, raw_depth_mm, intrinsic_matrix):
        """
        Converts 2D pixel coordinates (x, y) from a Realsense D435i frame
        to real-world (X, Y, Z) distances in meters.

        Args:
            x_pixel (int): The x-coordinate of the pixel.
            y_pixel (int): The y-coordinate of the pixel.
            raw_depth_mm (int or float): The raw depth value at the given pixel from the depth frame,
                                        expected to be in millimeters (mm).
            intrinsic_matrix (np.array): The camera's intrinsic matrix, typically 3x3.
                                        Example: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

        Returns:
            tuple: A tuple (X, Y, Z) representing the real-world coordinates in meters.
                X: Real-world X-coordinate (horizontal)
                Y: Real-world Y-coordinate (vertical)
                Z: Real-world Z-coordinate (depth)
        """

        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        # Convert raw depth from millimeters to meters
        Z_meters = raw_depth_mm / 1000.0

        # Convert pixel coordinates to normalized image plane coordinates
        x_normalized = (x_pixel - cx) / fx
        y_normalized = (y_pixel - cy) / fy

        # Scale by depth to get real-world X and Y in meters
        X_meters = x_normalized * Z_meters
        Y_meters = y_normalized * Z_meters

        return X_meters, Y_meters, Z_meters

    def run(self):
        while not rospy.is_shutdown():

            color_image_copy = self.color_image.copy()

            detected_boxes = []
            if self.yolo_enabled:
                results = self.model(color_image_copy, verbose=False)
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        detected_boxes.append((x1, y1, x2, y2))
                        cv2.rectangle(color_image_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if self.clicked_point:
                cx = self.clicked_point.x
                cy = self.clicked_point.y
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

                # cv2.imshow("masked_depth_img", masked_depth)
            
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
                white_mask_path = os.path.join(output_dir, 'white_mask.png')

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

                # Create a 480x640 one-channel image with all white pixels (255) and save it
                white_image = np.ones((480, 640), dtype=np.uint8) * 255
                cv2.imwrite(white_mask_path, white_image)

                rospy.loginfo(f"Color image saved to: {color_path}")
                rospy.loginfo(f"Depth image saved to: {depth_path}")
                rospy.loginfo(f"CameraInfo saved to: {meta_path}")

                # Calculate Object Location based on Camera Frame
                intrinsic_matrix = np.array([
                    [fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1]
                ], dtype=np.float32)
                cbbox_pixel_x = x + (w/2)
                cbbox_pixel_y = y + (h/2)
                raw_depth = masked_depth[int(cbbox_pixel_y), int(cbbox_pixel_x)]
                cbbox_real_x, cbbox_real_y, cbbox_real_z = self.pixel_to_real_world_coordinates(
                    cbbox_pixel_x, cbbox_pixel_y, raw_depth,
                    intrinsic_matrix
                )
                print(f"BB Pixel ({cbbox_pixel_x}, {cbbox_pixel_y}) with raw depth {raw_depth} mm:")
                print(f"BB Real-world coordinates (X, Y, Z): ({cbbox_real_x:.4f} m, {cbbox_real_y:.4f} m, {cbbox_real_z:.4f} m)")

                # Call detect_grasp service
                rospy.wait_for_service("detect_grasp")
                try:
                    detect_grasp = rospy.ServiceProxy("detect_grasp", GraspDetection)
                    res = detect_grasp(Point(x=cbbox_real_x, y=cbbox_real_y, z=cbbox_real_z))
                    print(f"Score: {res.score}")
                    print(f"Width: {res.width}")
                    print(f"Height: {res.height}")
                    print(f"Depth: {res.depth}")
                    print(f"Position: {res.position}")
                    print(f"Quaternion: {res.orientation}")

                    # Publish grasp pose data to related topics
                    self.grasp_score_pub.publish(Float32(data=res.score))
                    self.grasp_width_pub.publish(Float32(data=res.width))
                    self.grasp_height_pub.publish(Float32(data=res.height))
                    self.grasp_depth_pub.publish(Float32(data=res.depth))
                    self.grasp_pose_pub.publish(Pose(position=res.position, orientation=res.orientation))
                    # self.bbox_real_pub.publish(Point(x=cbbox_real_x, y=cbbox_real_y, z=cbbox_real_z))

                except rospy.ServiceException as e:
                    print("Service call failed: %s" % e)

                # break
                self.bbox = (0, 0, 0, 0)
                self.yolo_enabled_pub.publish(Bool(data=False))

            image_msg = self.bridge.cv2_to_imgmsg(color_image_copy, encoding='bgr8')
            self.image_pub.publish(image_msg)
            # cv2.imshow("Image Saver Masked Window", color_image_copy)
            # cv2.imshow("depth_img", self.depth_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            self.rate.sleep()
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    my_realsense = MyRealsense()
    my_realsense.run()