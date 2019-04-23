#!/usr/bin/env python
"""
This example shows an optimizer working with a set of n cameras, changing their pose so that the reprojection error is
minimized.
The OCDatasetLoader is used to collect data from a OpenConstructor dataset.
"""

# -------------------------------------------------------------------------------
# --- IMPORTS (standard, then third party, then my own modules)
# -------------------------------------------------------------------------------
import OCDatasetLoader.OCDatasetLoader as OCDatasetLoader
import OCDatasetLoader.OCArucoDetector as OCArucoDetector
import KeyPressManager.KeyPressManager as KeyPressManager
import argparse
from copy import deepcopy

import rospy
import tf
import cv2 as cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

br = tf.TransformBroadcaster()
rgb_pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=10)
# TODO: check proper topic names
depth_pub = rospy.Publisher('/camera/depth/image', Image, queue_size=10)

dataset_cameras = None
camera_idx = 0
loop = True


# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------

def timerTFCallback(event):
    global br, dataset_cameras, camera_idx
    t = rospy.Time.now()

    trans = tuple(dataset_cameras.cameras[camera_idx].rgb.matrix[0:3, 3])
    matrix = deepcopy(dataset_cameras.cameras[camera_idx].rgb.matrix)
    matrix[0:3, 3] = [0, 0, 0]

    rot = tf.transformations.quaternion_from_matrix(matrix)
    br.sendTransform(trans, rot, t, "/camera_rgb_optical_frame", "/map")

    trans = tuple(dataset_cameras.cameras[camera_idx].depth.matrix[0:3, 3])
    matrix = deepcopy(dataset_cameras.cameras[camera_idx].depth.matrix)
    matrix[0:3, 3] = [0, 0, 0]

    rot = tf.transformations.quaternion_from_matrix(matrix)
    br.sendTransform(trans, rot, t, "/camera_depth_optical_frame", "/map")


def counterCallback(event):
    global camera_idx

    if camera_idx + 1 >= len(dataset_cameras.cameras):
        if loop:
            camera_idx = 0
        else:
            rospy.signal_shutdown("Finished visiting all cameras.")
    else:
        camera_idx = camera_idx + 1
        rospy.loginfo("Publishing camera " + str(camera_idx))


def timerImagesCallback(event):
    global rgb_pub, dataset_cameras, camera_idx
    t = rospy.Time.now()

    try:
        msg = CvBridge().cv2_to_imgmsg(dataset_cameras.cameras[camera_idx].rgb.image, "bgr8")
    except CvBridgeError as e:
        print(e)  # TODO launch exception

    msg.header.frame_id = "/camera_rgb_optical_frame"
    msg.header.stamp = t

    rgb_pub.publish(msg)

    tmp = dataset_cameras.cameras[camera_idx].rgb.range_dense * 1000.0  # to millimeters
    tmp = tmp.astype(np.uint16)

    # tmp = dataset_cameras.cameras[camera_idx].rgb.range_dense
    # tmp = tmp.astype(np.float32)

    try:
        msg = CvBridge().cv2_to_imgmsg(tmp, "16UC1")  # or mono16?
        # msg = CvBridge().cv2_to_imgmsg(tmp, "32FC1")  # or mono16?
    except CvBridgeError as e:
        print(e)  # TODO launch exception

    msg.header.frame_id = "/camera_depth_optical_frame"
    msg.header.stamp = t

    depth_pub.publish(msg)


# -------------------------------------------------------------------------------
# --- MAIN
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    global dataset_cameras

    # ---------------------------------------
    # --- Parse command line argument
    # ---------------------------------------
    ap = argparse.ArgumentParser()

    # Dataset loader arguments
    ap.add_argument("-p", "--path_to_images", help="path to the folder that contains the OC dataset", required=True)
    ap.add_argument("-o", "--path_to_output_dataset", help="path to the folder that will contain the output OC dataset",
                    type=str, default=None, required=False)
    ap.add_argument("-ext", "--image_extension", help="extension of the image files, e.g., jpg or png", default='jpg')
    ap.add_argument("-m", "--mesh_filename", help="full filename to input obj file, i.e. the 3D model", required=True)
    ap.add_argument("-i", "--path_to_intrinsics", help="path to intrinsics yaml file", required=True)
    ap.add_argument("-ucci", "--use_color_corrected_images", help="Use previously color corrected images",
                    action='store_true', default=False)
    ap.add_argument("-si", "--skip_images", help="skip images. Useful for fast testing", type=int, default=1)
    ap.add_argument("-vri", "--view_range_image", help="visualize sparse and dense range images", action='store_true',
                    default=False)
    ap.add_argument("-ms", "--marker_size", help="Size in meters of the ArUco markers in the images", type=float,
                    required=True)
    ap.add_argument("-vad", "--view_aruco_detections", help="visualize ArUco detections in the camera images",
                    action='store_true',
                    default=False)
    ap.add_argument("-va3d", "--view_aruco_3d", help="visualize ArUco detections in a 3d window", action='store_true',
                    default=False)
    ap.add_argument("-va3dpc", "--view_aruco_3d_per_camera",
                    help="visualize all ArUco detections in a 3D window (plot becomes quite dense)",
                    action='store_true',
                    default=False)
    # OptimizationUtils arguments
    ap.add_argument("-sv", "--skip_vertices", help="skip vertices. Useful for fast testing", type=int, default=1)
    ap.add_argument("-z", "--z_inconsistency_threshold", help="threshold for max z inconsistency value", type=float,
                    default=0.05)
    ap.add_argument("-vpv", "--view_projected_vertices", help="visualize projections of vertices onto images",
                    action='store_true', default=False)
    ap.add_argument("-vo", "--view_optimization", help="...", action='store_true', default=False)
    # TODO create loop argument

    args = vars(ap.parse_args())
    print(args)

    # ---------------------------------------
    # --- INITIALIZATION
    # ---------------------------------------
    dataset_loader = OCDatasetLoader.Loader(args)
    dataset_cameras = dataset_loader.loadDataset()
    print("dataset_cameras contains " + str(len(dataset_cameras.cameras)) + " cameras")

    aruco_detector = OCArucoDetector.ArucoDetector(args)
    dataset_arucos, dataset_cameras = aruco_detector.detect(dataset_cameras)

    rospy.init_node('playback')
    rospy.Timer(rospy.Duration(0.1), timerTFCallback)
    rospy.Timer(rospy.Duration(0.3), timerImagesCallback)
    rospy.Timer(rospy.Duration(.5), counterCallback)

    #TODO publish camera_info
    #https://gist.github.com/rossbar/ebb282c3b73c41c1404123de6cea4771

    #TODO publish point clouds

    #TODO plugin com o chisel

    rospy.spin()

    # ---------------------------------------
    # --- PLAYBACK OC DATASET
    # ---------------------------------------

    for camera in dataset_cameras.cameras:
        # STEP1: publish of tf

        # STEP2: publish rgb image

        # STEP3: publish depth image

        pass
