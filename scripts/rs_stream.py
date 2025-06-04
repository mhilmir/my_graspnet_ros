#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

latest_frame = None  # Global variable to store the latest frame

def image_callback(msg):
    global latest_frame
    bridge = CvBridge()
    try:
        latest_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        rospy.logerr(f"Failed to convert image: {e}")

def main():
    global latest_frame
    rospy.init_node('image_listener', anonymous=True)
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if latest_frame is not None:
            cv2.imshow("Camera Frame", latest_frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key to exit
                break
        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
