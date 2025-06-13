# coding=utf-8

"""
Eye-to-hand calibration: compute the rotation and translation from the camera coordinate system
to the robot base coordinate system using images and corresponding robot poses.

"""

import os
import logging
import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from libs.auxiliary import find_latest_data_folder
from libs.log_setting import CommonLog

from save_poses2 import poses2_main

np.set_printoptions(precision=8,suppress=True)

logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

#setup path
current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"eye_hand_data")

images_path = os.path.join("eye_hand_data",find_latest_data_folder(current_path))

file_path = os.path.join(images_path,"poses.txt")  #When capturing images of the calibration board, 
#the corresponding poses of the robot's end-effector must match the order of the images — from the first line to the last line

#read checkerboard parameters from config.yaml
with open("config.yaml", 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

XX = data.get("checkerboard_args").get("XX") 
YY = data.get("checkerboard_args").get("YY") 
L = data.get("checkerboard_args").get("L")   

def func():

    path = os.path.dirname(__file__)
    print(path)

    # Set the parameters for finding sub-pixel corner points, using the stopping criteria of a maximum of 30 iterations and an error tolerance of 0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # Get the positions of the calibration board's corner points.
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)     # Set the world coordinate system on the calibration board — since all points lie on a plane, 
                                                            #their Z-coordinates are all 0, so only the x and y values need to be assigned
    objp = L*objp

    obj_points = []     # Store 3D points
    img_points = []     # Store 2D points

    images_num = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

    for i in range(1, len(images_num) + 1):   #The calibrated images are located in the images_path directory, named from 0.jpg to x.jpg.

        image_file = os.path.join(images_path,f"{i}.jpg")

        if os.path.exists(image_file):

            logger_.info(f'读 {image_file}')

            img = cv2.imread(image_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)

            if ret:

                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # Refine the original corner points to sub-pixel accuracy based on the initial detected corners.
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

    N = len(img_points)

    # Calibrate to obtain the pose of the checkerboard in the camera coordinate system.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    # logger_.info(f"内参矩阵:\n:{mtx}" ) # 内参数矩阵
    # logger_.info(f"畸变系数:\n:{dist}")  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

    print("-----------------------------------------------------")

    poses2_main(file_path)
    # save the pose of the robot's end-effector in the base coordinate system

    csv_file = os.path.join(path,"RobotToolPose.csv")
    tool_pose = np.loadtxt(csv_file,delimiter=',')

    R_tool = []
    t_tool = []

    for i in range(int(N)):

        R_tool.append(tool_pose[0:3,4*i:4*i+3])
        t_tool.append(tool_pose[0:3,4*i+3])

    R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)

    return R,t

if __name__ == '__main__':

    # 旋转矩阵
    rotation_matrix, translation_vector = func()

    # 将旋转矩阵转换为四元数
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    x, y, z = translation_vector.flatten()

    logger_.info(f"旋转矩阵是:\n {            rotation_matrix}")

    logger_.info(f"平移向量是:\n {            translation_vector}")

    logger_.info(f"四元数是：\n {             quaternion}")

