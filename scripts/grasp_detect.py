#!/usr/bin/env python3

import os
import sys
import rospy
import numpy as np
import open3d as o3d
import scipy.io as scio
# from PIL import Image
from PIL import Image as PILImage

import torch
from graspnetAPI import GraspGroup

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
    # color = np.array(Image.open(os.path.join(data_dir, 'color_ros.png')), dtype=np.float32) / 255.0
    color = np.array(PILImage.open(os.path.join(data_dir, 'color_ros.png')), dtype=np.float32) / 255.0

    # Load depth image (as .npy)
    depth = np.load(os.path.join(data_dir, 'depth_ros.npy'))

    # Load workspace mask and convert to boolean
    # workspace_mask = np.array(Image.open(os.path.join(data_dir, 'white_mask.png')))
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
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

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

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
