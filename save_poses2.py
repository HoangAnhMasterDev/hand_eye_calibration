"""
In an eye-to-hand setup, what we compute is the homogeneous transformation matrix from the camera to the robot base.

The transformation from the robot base to the robot end-effector is equal to the inverse of the transformation,
from the robot end-effector to the base (which is what we get from the robot’s pose data).

This code does:

    1. Reads the robot poses from a text file (poses.txt),

    2. Converts those poses (position + Euler angles) into homogeneous transformation matrices,

    3. Inverts those matrices to get the robot base to end-effector transform (as needed),

    4. And saves all these matrices into a CSV file (RobotToolPose.csv).


"""

import csv
import numpy as np


# Open text file containing the robot poses
def poses2_main(tag):


    with open(f"{tag}", "r",encoding="utf-8") as f:
        # 读取文件中的所有行
        lines = f.readlines()
    # 定义一个空列表，用于存储结果

    # 遍历每一行数据
    lines = [float(i)  for line in lines for i in line.split(',')]

    matrices = []

    for i in range(0,len(lines),6):
        matrices.append(inverse_transformation_matrix(pose_to_homogeneous_matrix(lines[i:i+6])))


    # 将齐次变换矩阵列表存储到 CSV 文件中
    save_matrices_to_csv(matrices, f'RobotToolPose.csv')

def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz@Ry@Rx  # 先绕 x轴旋转 再绕y轴旋转  最后绕z轴旋转
    return R


def pose_to_homogeneous_matrix(pose):

    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]

    return H

def inverse_transformation_matrix(T):
    R = T[:3, :3]
    t = T[:3, 3]

    # 计算旋转矩阵的逆矩阵
    R_inv = R.T

    # 计算平移向量的逆矩阵
    t_inv = -np.dot(R_inv, t)

    # 构建逆变换矩阵
    T_inv = np.identity(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv


def save_matrices_to_csv(matrices, file_name):
    rows, cols = matrices[0].shape
    num_matrices = len(matrices)
    combined_matrix = np.zeros((rows, cols * num_matrices))

    for i, matrix in enumerate(matrices):
        combined_matrix[:, i * cols: (i + 1) * cols] = matrix

    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in combined_matrix:
            csv_writer.writerow(row)

if __name__ == "__main__":
    # 假设已经将位姿列表转换为齐次变换矩阵列表
    # 示例：
    tag = ''
    poses2_main(tag)