#!/usr/bin/env python3

import os
import sys
import rospy
import numpy as np
import open3d as o3d
import scipy.io as scio
from PIL import Image as PILImage

import torch
from graspnetAPI import GraspGroup
from graspnetAPI.grasp import Grasp

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Path setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '../graspnet_lib/models'))
sys.path.append(os.path.join(ROOT_DIR, '../graspnet_lib/dataset'))
sys.path.append(os.path.join(ROOT_DIR, '../graspnet_lib/utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

def get_net(checkpoint_path, num_view):
    net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02,
                   hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    rospy.loginfo("Loaded checkpoint %s (epoch %d)", checkpoint_path, checkpoint['epoch'])
    net.eval()
    return net

def get_and_process_data(data_dir, factor_depth, width, height, num_point):
    # Load color image (normalized to [0, 1])
    color = np.array(PILImage.open(os.path.join(data_dir, 'color_ros.png')), dtype=np.float32) / 255.0

    # Load depth image (as .npy)
    depth = np.load(os.path.join(data_dir, 'depth_ros.npy'))

    # Load workspace mask and convert to boolean
    workspace_mask = np.array(PILImage.open(os.path.join(data_dir, 'white_mask.png')))
    workspace_mask = workspace_mask > 0

    # Load camera intrinsics
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']

    # Create camera info
    camera = CameraInfo(width, height,
                        intrinsic[0][0], intrinsic[1][1],
                        intrinsic[0][2], intrinsic[1][2],
                        factor_depth)

    # Generate organized point cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # Mask valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # Sample points
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # Convert to Open3D point cloud for visualization
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    # Convert sampled data to tensor
    cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Pack data for network
    end_points = {
        'point_clouds': cloud_sampled_tensor,
        'cloud_colors': color_sampled
    }

    return end_points, cloud_o3d

def get_grasps(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    return GraspGroup(gg_array)

def collision_detection(gg, cloud_np, voxel_size, thresh):
    mfcdetector = ModelFreeCollisionDetector(cloud_np, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=thresh)
    return gg[~collision_mask]

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:5]
    print(gg[0])
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def rpy_to_rotation_matrix(roll, pitch, yaw):
    """Create a rotation matrix from roll, pitch, yaw (Z-Y-X convention)"""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx  # ZYX order

def rotate_grasp_rotation_matrix(grasp_rotation_matrix, roll=0, pitch=0, yaw=0, local=True):
    """
    Rotate a 3x3 grasp rotation matrix with roll, pitch, yaw angles (in radians).
    - If local=True: rotates in the local object frame (post-multiplication)
    - If local=False: rotates in the world/global frame (pre-multiplication)
    """
    R_rpy = rpy_to_rotation_matrix(roll, pitch, yaw)
    if local:
        return grasp_rotation_matrix @ R_rpy
    else:
        return R_rpy @ grasp_rotation_matrix

def update_grasp_rotation(grasp: Grasp, roll, pitch, yaw, local=True):
    # Get original rotation matrix
    R_orig = grasp.rotation_matrix  # shape (3, 3)
    R_new = rotate_grasp_rotation_matrix(R_orig, roll, pitch, yaw, local=local)
    R_flat = R_new.flatten()

    # Copy original grasp array and replace the rotation part
    new_grasp_array = grasp.grasp_array.copy()
    new_grasp_array[:9] = R_flat  # Assuming first 9 elements are rotation

    return Grasp(new_grasp_array)

def main():
    rospy.init_node("grasp_detect_node", anonymous=True)

    # Read parameters from launch file
    checkpoint_path = rospy.get_param("~checkpoint_path")
    num_point = rospy.get_param("~num_point", 20000)
    num_view = rospy.get_param("~num_view", 300)
    collision_thresh = rospy.get_param("~collision_thresh", 0.01)
    voxel_size = rospy.get_param("~voxel_size", 0.01)
    factor_depth = rospy.get_param("~factor_depth", 1000.0)
    image_width = rospy.get_param("~image_width", 640.0)
    image_height = rospy.get_param("~image_height", 480.0)
    data_dir = rospy.get_param("~data_dir")

    rospy.loginfo("Starting Grasp Detection Node...")
    rospy.loginfo("Loading network...")
    net = get_net(checkpoint_path, num_view)

    rospy.loginfo("Processing data...")
    end_points, cloud = get_and_process_data(data_dir, factor_depth, image_width, image_height, num_point)

    rospy.loginfo("Generating grasps...")
    gg = get_grasps(net, end_points)

    if collision_thresh > 0:
        rospy.loginfo("Performing collision detection...")
        gg = collision_detection(gg, np.array(cloud.points), voxel_size, collision_thresh)

    rospy.loginfo("Visualizing grasps...")
    vis_grasps(gg, cloud)

    # PERCOBAAN MEROTASI GRASPING POSE
    gg.nms()
    gg.sort_by_score()
    # g = gg[0]

    # print(type(gg[0]))
    # print(dir(gg[0]))
    print(gg[0].grasp_array.shape)  # should print (12,)

    rospy.loginfo("Visualizing grasps..., [BEFORE rotation]")
    # vis_grasps(gg[:1], cloud)
    print(gg[0].rotation_matrix)

    # Apply a 30-degree roll
    roll = 0
    pitch = np.radians(90)
    yaw = np.radians(90)
    # Rotate
    # gg[0].rotation_matrix = rotate_grasp_rotation_matrix(gg[0].rotation_matrix, roll, pitch, yaw, local=True)
    # result = rotate_grasp_rotation_matrix(gg[0].rotation_matrix, roll, pitch, yaw, local=True)
    gg.grasp_group[0] = update_grasp_rotation(gg.grasp_group[0], roll, pitch, yaw, local=True)

    rospy.loginfo("Visualizing grasps..., [AFTER rotation]")
    # vis_grasps(gg[:1], cloud)
    print(gg[0].rotation_matrix)
    # print(result)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
