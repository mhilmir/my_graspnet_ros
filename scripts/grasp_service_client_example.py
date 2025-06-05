#!/usr/bin/env python3

import rospy
from my_graspnet_ros.srv import GraspDetection
from geometry_msgs.msg import Point, Quaternion

rospy.init_node("grasp_client")

rospy.wait_for_service("detect_grasp")
try:
    detect_grasp = rospy.ServiceProxy("detect_grasp", GraspDetection)
    res = detect_grasp()
    print(f"Score: {res.score}")
    print(f"Position: {res.position}")
    print(f"Quaternion: {res.orientation}")
except rospy.ServiceException as e:
    print("Service call failed: %s" % e)