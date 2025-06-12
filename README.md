## Overview

Hand-eye calibration is commonly used in robotics and computer vision, especially in scenarios where precise control of a robotic arm interacting with the environment is required. Hand-eye calibration aligns the coordinate systems of the robotic arm (the "hand") and the camera (the "eye"), solving the transformation relationship between the two so that the robotic arm can accurately grasp targets identified by the camera.

The hand-eye system refers to the relationship between the hand (robotic arm) and the eye (camera). When the camera sees an object, it needs to inform the robotic arm where the object is located. Once the object's position is determined in the camera's coordinate system, and if the relationship between the camera and the robotic arm is known, the object's position in the robotic arm's coordinate system can be obtained.

Hand-eye calibration is essential in various scenarios such as robotic object grasping, interaction in dynamic environments, precision measurement and inspection, and visual servo control.

Once hand-eye calibration is completed, it does not need to be redone unless there is a change in the relative position between the camera and the robotic arm. The following calibration methods are applicable to upright, side-mounted, and inverted configurations.



There are two types of hand-eye calibration scenarios:

One is when the camera is mounted on the robotic arm.

Eye-in-Hand System: In this configuration, the camera is installed at the end of the robotic arm and moves along with the arm during its motion.
![image](picture/f6c716fb-c8d2-4adc-b3da-a86c6b1e78d0.png)



The second type is when the camera is fixed somewhere outside the robotic arm.

Eye-to-Hand System: In this configuration, the camera is mounted in a fixed position in the environment, separate from the robotic arm. It observes the workspace from a static viewpoint, and the robotic arm moves independently within the camera's field of view

  ​	![44776e79-47f7-4de2-9ef2-172b654169d5](picture/44776e79-47f7-4de2-9ef2-172b654169d5-17291349013411.png)



## HOW?

### 1.眼在手上（eye-in-hand)

​		**For the Eye-in-Hand configuration, hand-eye calibration refers to determining the coordinate transformation between the camera and the end-effector (the end of the robotic arm).:**

This transformation enables the system to convert the position of an object detected by the camera into the robotic arm’s coordinate system, allowing accurate and coordinated interaction with the environment.

![image](picture/f6c716fb-c8d2-4adc-b3da-a86c6b1e78d0.png)

When transforming the 3D spatial coordinates of a target point, the first challenge is determining the position and orientation relationship between the robot end-effector and the camera coordinate system. This relationship is the core result of hand-eye calibration and is denoted by the transformation matrix X. It can be solved using the equation AX = XB:

- A represents the transformation of the robot end-effector between two different poses (i.e., the relative motion of the end-effector).

- B represents the relative motion of the camera (i.e., how the camera sees its own movement between the same two poses).


As shown in Figure 1, this setup is for the Eye-in-Hand configuration. (Note: the figure is for illustration purposes only and not from a real experimental setup.) In this setup, the camera is mounted at the end of the robotic arm and moves along with it。



![图1 眼在手上](picture/1b3bb9f5348fe9f1dd4ae02afed614e9.png)

- A：The pose of the robot end-effector in the robot’s base coordinate system can be obtained through the robot’s API.。

$$ {}^{base}_{end}M $$
  
- B：The pose of the camera in the robot end-effector coordinate system is a fixed transformation, meaning it does not change as long as the camera remains rigidly mounted on the end-effector.

  Once this transformation is known, we can compute the camera’s actual position and orientation at any time based on the end-effector's pose.

$$ {}^{end}_{camera}M $$
  
- C：The pose of the camera with respect to the calibration board coordinate system is essentially the camera's extrinsic parameters—that is, the position and orientation of the camera relative to the calibration board。

  This transformation is obtained through camera calibration, specifically by using known patterns (such as a checkerboard) and applying standard extrinsic calibration techniques.
  
$$ {}^{board}_{camera}M $$

- D：The pose of the calibration board with respect to the robot base coordinate system is fixed during the calibration process.。


$$ {}^{base}_{board}M $$

So as long as we can calculate the transformation B, the pose D of the calibration board in the robot arm's coordinate system will naturally be obtained.:

$$ {}^{base}{board}M = {}^{base}{end}M \cdot {}^{end}{camera}M \cdot {}^{camera}{board}M $$

As shown in Figure 2, we move the robot arm to two different positions, ensuring that the calibration board is visible from both. Then, we construct a spatial transformation loop.：

![图2 机械臂运动到两个位置，构建变换回路](picture/29fb4d433468f12530eca3e2a563da72.png)

$$ A_1 \cdot B \cdot C_1^{-1} = A_2 \cdot B \cdot C_2^{-1} $$

$$ \left( A_2^{-1} \cdot A_1 \right) \cdot B = B \cdot \left( C_2^{-1} \cdot C_1 \right) $$

Which is equivalent to the following equation：


$$ {}^{base}{end}M_1 \cdot {}^{end}{camera}M_1 \cdot {}^{camera}{board}M_1 = {}^{base}{end}M_2 \cdot {}^{end}{camera}M_2 \cdot {}^{camera}{board}M_2 $$

$$ \left( {}^{base}{end}M_2 \right)^{-1} \cdot {}^{base}{end}M_1 \cdot {}^{end}{camera}M_1 = {}^{end}{camera}M_2 \cdot {}^{camera}{board}M_2 \cdot \left( {}^{camera}{board}M_1 \right)^{-1} $$

This is a classic **AX=XB** problem，and by definition，X is a 4×4 homogeneous transformation matrix：

$$ X = \begin{bmatrix} R & t \\\ 0 & 1 \end{bmatrix} $$

The goal of hand-eye calibration is to compute X

​	
### 2 .眼在手外（eye-to-hand)

### ![44776e79-47f7-4de2-9ef2-172b654169d5](picture/44776e79-47f7-4de2-9ef2-172b654169d5-17291482180503.png)

**During eye-to-hand calibration, the robot base and the camera are fixed, and the calibration board is mounted on the robot's end-effector. Therefore, during the calibration process, the relationship between the calibration board and the end-effector remains constant, as does the relationship between the camera and the robot base.**

The goal of the calibration is to determine the transformation matrix from the camera coordinate system to the robot base coordinate system.

$$ {}^{base}_{camera}M $$

Implementation steps：

          1.Fix the calibration board to the end-effector of the robot arm.

​					2.Move the end-effector and use the camera to capture n images (typically 10–20) of the calibration board at different robot poses.

For each image and corresponding robot pose, the following equation holds:：

$$ {}^{end}{board}M = {}^{end}{base}M \cdot {}^{base}{camera}M \cdot {}^{camera}{board}M $$


where：

| Symbol               | Description                         |
| ------------------ | ---------------------------- |
|$$^{end}_{board}M$$  | The transformation matrix from the calibration board to the robot arm’s end-effector (since during calibration the calibration board is fixed to the end-effector, this transformation matrix remains constant). |
|$${}^{end}_{base}M$$ | It can be calculated from the pose of the robot arm’s end-effector.  |
|$${}^{base}_{camera}M$$ | What hand-eye calibration needs to solve for             |
|$${}^{camera}_{board}M$$ | Obtained through camera calibration methods         |



Then the following equation can be obtained:：

**The Cauchy-Schwarz Inequality**

$$ {}^{end}{base}M_1 \cdot {}^{base}{camera}M_1 \cdot {}^{camera}{board}M_1 = {}^{end}{base}M_2 \cdot {}^{base}{camera}M_2 \cdot {}^{camera}{board}M_2 $$

$$ {}^{end}{base}M_2^{-1} \cdot {}^{end}{base}M_1 \cdot {}^{base}{camera}M_1 = {}^{base}{camera}M_2 \cdot {}^{camera}{board}M_2 \cdot {}^{camera}{board}M_1^{-1} $$

$$ \vdots $$

$$ {}^{end}{base}M_n^{-1} \cdot {}^{end}{base}M_{n-1} \cdot {}^{base}{camera}M{n-1} = {}^{base}{camera}M_n \cdot {}^{camera}{board}M_n \cdot {}^{camera}{board}M{n-1}^{-1} $$



This is also a classic **AX=XB** problem. By definition, X is a 4×4 homogeneous transformation matrix, where R is the rotation matrix from the camera to the robot base coordinate system, and t is the translation vector from the camera to the robot base coordinate system.：

$$ X = \begin{bmatrix} R & t \\\ 0 & 1 \end{bmatrix} $$

The purpose of hand-eye calibration is to calculate X。



## Key code explanation

### Code structure

```
---eye_hand_data  Data collected during eye-in-hand calibration

---libs

​         ---auxiliary.py Here are some auxiliary packages used in the program

​         ---log_settings.py Logging package

---robotic_arm_package Robot arm Python package

---collect_data.py Data collection program for eye-in-hand calibration

---compute_in_hand.py Calculation program for eye-in-hand calibration

---compute_to_hand.py Calculation program for eye-to-hand calibration

---requirements.txt Environment dependency files

---save_poses.py Computation dependency files

---save_poses2.py Computation dependency files
```



### compute_in_hand.py | compute_to_hand.py EXPLAIN

#### 1.主函数`func()`

**compute_in_hand.py | compute_to_hand.py**

```python
def func():
    
    path = os.path.dirname(__file__)

    # 亚像素角点查找准则
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 准备标定板的3D点坐标
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)
    objp = L * objp

    obj_points = []     # 存储3D点
    img_points = []     # 存储2D点

    images_num = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

    for i in range(1, len(images_num) + 1):
        image_file = os.path.join(images_path, f"{i}.jpg")

        if os.path.exists(image_file):
            logger_.info(f'读 {image_file}')

            img = cv2.imread(image_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)

            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

```



上面代码遍历采集的标定板图像，逐一获取棋盘格角点并放到数组中去

#### 2.相机标定

**compute_in_hand.py | compute_to_hand.py**

```python


# 标定，得到图案在相机坐标系下的位姿
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

```



- 使用OpenCV的 `calibrateCamera` 函数进行相机标定，计算相机的内参数和畸变系数。

- `rvecs` 和 `tvecs` 分别是每张图像的旋转向量和平移向量，表示**标定板在相机坐标系下的位姿**。

  

#### 3.处理机械臂位姿数据

**compute_in_hand.py**

```python
poses_main(file_path)
```

将机械臂末端位姿数据转化为**机械臂末端坐标系**相对于**基坐标系**的旋转矩阵和平移向量

**compute_to_hand.py**

```python
poses2_main(file_path)
```

将机械臂末端位姿数据转化为**基坐标系**相对于**机械臂末端坐标系**的旋转矩阵和平移向量



####  4.手眼标定计算

```python
R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)

return R,t

```

- 使用OpenCV的 `calibrateHandEye` 函数进行手眼标定

  - 在compute_in_hand.py中计算**相机**相对于**机械臂末端**的旋转矩阵 **R_cam2end**和平移向量 **T_cam2end**。

    ```
    void
    cv::calibrateHandEye(InputArrayOfArrays 	R_end2base,
                         InputArrayOfArrays 	T_end2base, 
                         InputArrayOfArrays 	R_board2cam,
                         InputArrayOfArrays 	T_board2cam,
                         OutputArray 	        R_cam2end,
                         OutputArray 	        T_cam2end, 
                         HandEyeCalibrationMethod method = CALIB_HAND_EYE_TSAI)	
    
    ```

    

  - 在compute_to_hand.py中计算**相机**相对于**机械臂基座**的旋转矩阵 **R_cam2base** 和平移向量**T_cam2base**。

    ```
    void
    cv::calibrateHandEye(InputArrayOfArrays 	R_base2end
                         InputArrayOfArrays 	T_base2end
                         InputArrayOfArrays 	R_board2cam
                         InputArrayOfArrays 	T_board2cam
                         OutputArray 	        R_cam2base
                         OutputArray 	        T_cam2base
                         HandEyeCalibrationMethod method = CALIB_HAND_EYE_TSAI)	
    
    ```

    

- 采用了 `CALIB_HAND_EYE_TSAI` 方法，这是常用的手眼标定算法



### `save_poses.py` 关键代码解释

#### 1. **定义欧拉角转换为旋转矩阵的函数**

```python
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

    R = Rz @ Ry @ Rx

    return R

```

- 定义了将欧拉角（绕X、Y、Z轴的旋转）转换为旋转矩阵的函数。
- 按照Z-Y-X的顺序进行旋转，并将各轴的旋转矩阵相乘得到最终的旋转矩阵 `R`。

#### 2. **将位姿转换为齐次变换矩阵**

```python
def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]

    return H

```

- 将位姿（位置和姿态）转换为4x4的齐次变换矩阵，以便进行矩阵运算。
- 位置 `(x, y, z)` 作为平移向量，姿态 `(rx, ry, rz)` 作为旋转欧拉角。
- 生成的齐次变换矩阵 `H` 将用于描述**机械臂末端相对于基座的旋转变换**。

#### 3. **保存多个矩阵到CSV文件**

```python
Copy codedef save_matrices_to_csv(matrices, file_name):
    rows, cols = matrices[0].shape
    num_matrices = len(matrices)
    combined_matrix = np.zeros((rows, cols * num_matrices))

    for i, matrix in enumerate(matrices):
        combined_matrix[:, i * cols: (i + 1) * cols] = matrix

    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in combined_matrix:
            csv_writer.writerow(row)
```

- 将多个矩阵横向拼接成一个大矩阵，然后保存到CSV文件中。

- 每个矩阵占用固定的列数，这样在读取时可以按照固定的间隔切分出单个矩阵。

  

## 标定过程

### 1.环境要求

#### 基础环境准备

| 项目     | 版本           |
| :------- | :------------- |
| 操作系统 | ubuntu/windows |
| Python   | 3.9及以上      |
|          |                |

#### Python环境准备

| 包            | 版本        |
| :------------ | :---------- |
| numpy         | 2.0.2       |
| opencv-python | 4.10.0.84   |
| pyrealsense2  | 2.55.1.6486 |
| scipy         | 1.13.1      |

执行下面命令在python环境中安装手眼标定程序所需要的包：

```cmd
pip install -r requirements.txt
```



#### 设备准备

- 机械臂：RM75 RM65  RM63 GEN72 
- 相机: Intel RealSense Depth Camera D435

- 摄像头专用数据线

- 网线

- 标定板（1或2）

  1. 打印纸质标定板

     ![标定板图片_00(1)](picture/标定板图片_00(1).png)

  2. 淘宝上搜索“标定板棋盘格”购买







### 2.标定过程

#### 参数配置

在配置文件（config.yaml）中设置标定板参数

![image-20250403064543828](picture/image-20250403064543828.png)

​		

config.yaml里的配置参数如下，有下面三个

​        xx:标定板的横向角点数（长边格子数减1），默认为11，例如下图长边12个格子，角点数为11

​		YY:标定板纵向角点数（短边格子数减1），默认为8，例如下图短边9个格子，角点数为8

​		L :  标定板单个方格的实际尺寸（单位：米），默认是0.03

​        ![](picture/image-20241016181226851.png)



#### 眼在手上

##### 采集数据

(1).相机连接线连接**电脑**和D435相机，网线连接**电脑**和机械臂

(2).电脑的ip和机械臂设置为同一网段

如果机械臂IP为192.168.1.18 ，则将电脑 ip地址设为1网段

![image-20241018143659263](picture/image-20241018143659263.png)

​			如果机械臂的IP为192.168.10.18，则将电脑的IP设置为10网段（参照上面设置）                                     

(3).标定板放置在平面上，相机固定在机械臂末端，相机对准标定板

​		标定过程中，标定板是**固定放置**在机械臂工作区内的，这个固定位置需要保证安装在**机械臂末端的相机能从不同的视角**观测到它，标定板的确切位置不重要，因为不必知道其相对于机械臂基座的位姿。但是标定板在标定期间应**保持固定，不得移动**。

(4).运行脚本`collect_data.py`，出现一个弹窗

(5).拖动机械臂末端，使呈现相机视野下的标定板清晰、完整、将光标放在弹窗上

​		**注意：**

- 使出现在相机视野里的**标定板**和**相机镜面**呈现**一定角度**。

  ![image-20241101112801145](picture/image-20241101112801145.png)

  下面的拍照姿势**不对**：
  ![image-20241101112816914](picture/image-20241101112816914.png)

​				

(6).点击键盘“s”采集数据

(7)移动15-20次机械臂，重复步骤(5)(6)，采集不同机械臂姿态下的标定板图片15-20张左右

​	**注意：**

​			移动**机械臂末端旋转轴**，每次旋转的角度尽量的大（大于30°）

​			确保在三个轴（X、Y、Z）上都有足够的旋转角度变化。

​			（可以先绕着机械臂末端z轴旋转多个角度拍摄多副图片，然后绕着x轴旋转）

​			![WPS拼图1](picture/WPS拼图1.png)

##### 计算标定结果

​		运行脚本`compute_in_hand.py`，获取标定结果

​        得出**相机坐标系**相对于**机械臂末端**坐标系的**旋转矩阵**和**平移向量**

#### 眼在手外

##### 采集数据

(1).相机连接线连接**电脑**和D435相机，网线连接**电脑**和机械臂

(2).电脑的ip和机械臂设置为同一网段

​	如果机械臂IP为192.168.1.18 ，则将电脑 ip地址设为1网段

​    如果机械臂的IP为192.168.10.18，则将电脑的IP设置为10网段

(3).将标定板（**打印纸质较小的板子，方便固定**）**固定在机械臂末端，相机固定不动**，移动机械臂末端，使标定板出现在相机视野里

​       标定过程中，标定板安装在**机械臂末端执行器上**并随机械臂移动。可以直接固定在工具法兰上或由夹具固定安装，安装的确切位置不重要，因为**不必知道标定板和末端执行器**的相对位姿。重要的是标定板在运动过程中不会出现**相对于工具法兰或夹具的位移**，它必须被良好地固定住或被夹具紧紧地抓住。建议使用由刚性材料制成的安装支架。

(4).运行脚本`collect_data.py`，出现一个弹窗

(5).拖动机械臂末端，使呈现相机视野下的标定板清晰、完整、将光标放在弹窗上

​		**注意：**

​				使出现在相机视野里的标定板和相机镜面呈现一定角度。

(6).点击键盘“s”采集数据

(7)移动15-20次机械臂，重复步骤(5)(6)，采集不同机械臂姿态下的标定板图片15-20张左右

​			移动**机械臂末端旋转轴**，每次旋转的角度尽量的大（大于30°）

​			确保在三个轴（X、Y、Z）上都有足够的旋转角度变化。



##### 计算标定结果

运行脚本`compute_to_hand.py`，获取标定结果

得出**相机坐标系**相对于**机械臂基坐标系**的**旋转矩阵**和**平移向量**



**平移向量**的单位是米



#### 误差范围

受采集到的图片的质量影响，标定结果中的平移向量与实际的差距在1cm之内。

## 标定过程中可能出现的问题

### 问题1

在计算采集到的数据时

即执行下面脚本时：

```cmd
python compute_in_hand.py
```

或

```python
python compute_to_hand.py
```

可能出现下面问题

问题描述：[ERROR:0@1.418] global calibration_handeye.cpp:335 calibrateHandEyeTsai Hand-eye calibration failed! Not enough informative motions--include larger rotations.



**问题原因**：采集的图片旋转量不足，特别是缺少足够大的旋转运动。

手眼标定需要机器人在空间中执行一系列运动，以获取相机相对于机器人末端的精确关系。若机器人的运动数据中缺乏足够的旋转变化，尤其是在各个轴向上的显著旋转，标定算法就无法准确计算出手眼关系。



**解决方案：**

1. **增加旋转运动：**

- 在数据采集过程中，让机器人执行更大的旋转运动。确保在三个轴（X、Y、Z）上都有足够的旋转角度变化。
- 例如，旋转角度最好能超过30度，这样能提供丰富的运动信息。

2. **多样化运动姿态：**

- 采集数据时，保证机器人末端执行多样化的姿态变化，包括平移和旋转。
- 避免只在小范围内运动或只在某一轴上运动。

3. **增加数据采集次数：**

- 采集更多的样本数据，一般来说，至少需要10组以上不同的姿态数据。
- 更多的数据能提高标定的稳定性和准确性。

## 手眼标定结果如何使用



机械臂如何捡取物品？

### 眼在手上

#### 理论

从一个**不涉及相机**的机械臂开始。它的两个主要坐标系是：

​	1.机械臂基坐标系

   2.末端执行器坐标系

![../../../_images/hand-eye-robot-ee-robot-base-coordinate-systems.png](picture/hand-eye-robot-ee-robot-base-coordinate-systems-17301939637152.png)






为了捡取物品，机械臂需要知道**物品相对于机械臂基坐标系**的位姿。通过这些信息以及机器人相关的几何知识，即可算出末端执行器/夹具朝物体移动的关节角度。

![../../../_images/hand-eye-robot-robot-to-object.png](picture/hand-eye-robot-robot-to-object.png)

**物体相对相机坐标系的位姿**可以通过**模型识别**得到，为了使机械臂能够拾取物体，需要将物体的位姿从**相机的坐标系**转换到**机械臂的基坐标系**。

![../../../_images/hand-eye-robot-ee-robot-base-coordinate-systems-with-camera.png](picture/hand-eye-robot-ee-robot-base-coordinate-systems-with-camera.png)

在这种情况下，坐标转换是间接完成的：

$$ H^{ROB}{OBJ} = H^{ROB}{EE} \cdot H^{EE}{CAM} \cdot H^{CAM}{OBJ} $$

末端执行器相对于机械臂基座的位姿

$$H^{ROB}_{EE}$$

是已知的，通过机械臂API可以获取得到，相机相对于末端执行器的位姿

$$H^{EE}_{CAM}$$

由手眼标定得到。

![img](picture/hand-eye-eye-in-hand-all-poses.png)



假如物体在相机坐标系是一个3D点或者一个位姿，那么下面说明物体位置作为一个3D点或者一个位姿**从相机坐标系转换到机械臂基坐标系**的数学理论。

![../../../_images/hand-eye-eye-in-hand-all-poses.png](picture/hand-eye-eye-in-hand-all-poses-17302010389227.png)

机械臂的位姿用齐次变换矩阵表示。

以下方程描述了如何将单个3D点从相机坐标系转换到机械臂基坐标系：

$$ p^{ROB} = H^{ROB}{EE} \cdot H^{EE}{CAM} \cdot p^{CAM} $$

$$ \begin{bmatrix} x^r \\\ y^r \\\ z^r \\\ 1 \end{bmatrix} = \begin{bmatrix} R_e^r & t_e^r \\\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} R_c^e & t_c^e \\\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} x^c \\\ y^c \\\ z^c \\\ 1 \end{bmatrix} $$

如果要将物体位姿从相机坐标系转换到机械臂基坐标系。

$$ H^{ROB}{OBJ} = H^{ROB}{EE} \cdot H^{EE} {CAM} \cdot H^{CAM} {OBJ} $$

$$ \begin{bmatrix} R_o^r & t_o^r \\\ 0 & 1 \end{bmatrix} = \begin{bmatrix} R_e^r & t_e^r \\\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} R_c^e & t_c^e \\\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} R_o^c & t_o^c \\\ 0 & 1 \end{bmatrix} $$


​	由此产生的位姿是机械臂当前工具坐标系圆心应该达到的位姿进行捡取。(标定时采集的机械臂位姿也是当前工具坐标系相对于机械臂基坐标系位姿)



#### 代码



- 物体在相机坐标系是一个3D点（x,y,z)

  ```python
  
  import numpy as np
  from scipy.spatial.transform import Rotation as R
  
  
  # 相机坐标系到机械臂末端坐标系的旋转矩阵和平移向量（手眼标定得到）
  rotation_matrix = np.array([[-0.00235395 , 0.99988123 ,-0.01523124],
                              [-0.99998543, -0.00227965, 0.0048937],
                              [0.00485839, 0.01524254, 0.99987202]])
  translation_vector = np.array([-0.09321419, 0.03625434, 0.02420657])
  
  
  def convert(x ,y ,z ,x1 ,y1 ,z1 ,rx ,ry ,rz):
      """
      我们需要将手眼标定得到旋转向量和平移向量转换为齐次变换矩阵，然后使用深度相机识别到的物体坐标（x, y, z）和
      机械臂末端的位姿（x1,y1,z1,rx,ry,rz）来计算物体相对于机械臂基座的位姿（x, y, z）
  
      """
  
  
  
      # 深度相机识别物体返回的坐标
      obj_camera_coordinates = np.array([x, y, z])
  
      # 机械臂末端的位姿，单位为弧度
      end_effector_pose = np.array([x1, y1, z1,
                                    rx, ry, rz])
  
      # 将旋转矩阵和平移向量转换为齐次变换矩阵
      T_camera_to_end_effector = np.eye(4)
      T_camera_to_end_effector[:3, :3] = rotation_matrix
      T_camera_to_end_effector[:3, 3] = translation_vector
  
      # 机械臂末端的位姿转换为齐次变换矩阵
      position = end_effector_pose[:3]
      orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()
  
      T_base_to_end_effector = np.eye(4)
      T_base_to_end_effector[:3, :3] = orientation
      T_base_to_end_effector[:3, 3] = position
  
      # 计算物体相对于机械臂基座的位姿
      obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1])  # 将物体坐标转换为齐次坐标
  
      obj_end_effector_coordinates_homo = T_camera_to_end_effector.dot(obj_camera_coordinates_homo)
  
      obj_base_coordinates_homo = T_base_to_end_effector.dot(obj_end_effector_coordinates_homo)
  
      obj_base_coordinates = list(obj_base_coordinates_homo[:3])  # 从齐次坐标中提取物体的x, y, z坐标
  
  
      return obj_base_coordinates
  
  
  ```

  

- 物体在相机坐标系是一个位姿

  ```python
  
  import numpy as np
  from scipy.spatial.transform import Rotation as R
  
  
  # 相机坐标系到机械臂末端坐标系的旋转矩阵和平移向量（手眼标定得到）
  rotation_matrix = np.array([[-0.00235395 , 0.99988123 ,-0.01523124],
                              [-0.99998543, -0.00227965, 0.0048937],
                              [0.00485839, 0.01524254, 0.99987202]])
  translation_vector = np.array([-0.09321419, 0.03625434, 0.02420657])
  
  
  def decompose_transform(matrix):
      """
      将矩阵转化为位姿
      """
  
      translation = matrix[:3, 3]
      rotation = matrix[:3, :3]
  
      # Convert rotation matrix to euler angles (rx, ry, rz)
      sy = np.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
      singular = sy < 1e-6
  
      if not singular:
          rx = np.arctan2(rotation[2, 1], rotation[2, 2])
          ry = np.arctan2(-rotation[2, 0], sy)
          rz = np.arctan2(rotation[1, 0], rotation[0, 0])
      else:
          rx = np.arctan2(-rotation[1, 2], rotation[1, 1])
          ry = np.arctan2(-rotation[2, 0], sy)
          rz = 0
  
      return translation, rx, ry, rz
  
  
  def convert(x,y,z,rx,ry,rz,x1,y1,z1,rx1,ry1,rz1):
  
      """
  
      我们需要将标定得到的旋转向量和平移向量转换为齐次变换矩阵，然后使用机械臂末端的位姿(x, y, z,rx,ry,rz）和
     深度相机识别到的物体坐标（x1,y1,z1,rx1,ry1,rz1）来计算物体相对于机械臂基座的位姿
  
      """
  
  
  
      # 深度相机识别物体返回的坐标
      obj_camera_coordinates = np.array([x1, y1, z1,rx1,ry1,rz1])
  
      # 机械臂末端的位姿，单位为弧度
      end_effector_pose = np.array([x, y, z,
                                    rx, ry, rz])
  
      # 将旋转矩阵和平移向量转换为齐次变换矩阵
      T_camera_to_end_effector = np.eye(4)
      T_camera_to_end_effector[:3, :3] = rotation_matrix
      T_camera_to_end_effector[:3, 3] = translation_vector
  
      # 机械臂末端的位姿转换为齐次变换矩阵
      position = end_effector_pose[:3]
      orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()
  
      T_end_to_base_effector = np.eye(4)
      T_end_to_base_effector[:3, :3] = orientation
      T_end_to_base_effector[:3, 3] = position
  
      # 计算物体相对于机械臂基座的位姿
  
  
      # 物体相对于相机的位姿转换为齐次变换矩阵
      position2 = obj_camera_coordinates[:3]
      orientation2 = R.from_euler('xyz', obj_camera_coordinates[3:], degrees=False).as_matrix()
  
      T_object_to_camera_effector = np.eye(4)
      T_object_to_camera_effector[:3, :3] = orientation2
      T_object_to_camera_effector[:3, 3] = position2
  
  
      obj_end_effector_coordinates_homo = T_camera_to_end_effector.dot(T_object_to_camera_effector)
  
      obj_base_effector = T_end_to_base_effector.dot(obj_end_effector_coordinates_homo)
  
      result = decompose_transform(obj_base_effector)
  
  
  
      return result
  
  
  
  ```
  
  

​		   代码中rotation_matrix和translation_vector变量分别是眼在手上**手眼标定**得到的**旋转矩阵**和**平移向量**

### 眼在手外

#### 理论

相机可以通过模型获取物体在相机坐标系的里的位姿，**物体相对于机械臂的位姿**通过相机相对于机械臂基坐标系的位姿和物体相对于相机相机坐标系的位姿通过后乘法计算得到的：

$$ H^{ROB}{OBJ} = H^{ROB}{CAM} \cdot H^{CAM}{OBJ} $$

![../../../_images/hand-eye-eye-to-hand-all-poses.png](picture/hand-eye-eye-to-hand-all-poses.png)



假如物体位置是一个3D点或者一个位姿，那么下面说明物体位置作为一个3D点或者一个位姿**从相机坐标系转换到机械臂基坐标系**的数学理论。

以下方程描述了如何将单个3D点从相机坐标系转换到机械臂基坐标系：

$$ p^{ROB} = H^{ROB}{CAM} \cdot p^{CAM} $$

$$ \begin{bmatrix} x^r \\\ y^r \\\ z^r \\\ 1 \end{bmatrix} = \begin{bmatrix} R_c^r & t_c^r \\\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} x^c \\\ y^c \\\ z^c \ 1 \end{bmatrix} $$

如果要将物体位姿从相机坐标系转换到机械臂基坐标系。

$$ H^{ROB}{OBJ} = H^{ROB}{CAM} \cdot H^{CAM}{OBJ} $$

$$ \begin{bmatrix} R_o^r & t_o^r \\\ 0 & 1 \end{bmatrix} = \begin{bmatrix} R_c^r & t_c^r \\\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} R_o^c & t_o^c \\\ 0 & 1 \end{bmatrix} $$

#### 代码

- 物体在相机坐标系是一个3D点（x,y,z)

  ```python
  
  import numpy as np
  
  
  
  # 相机坐标系到机械臂基坐标系的旋转矩阵和平移向量(手眼标定得到)
  rotation_matrix = np.array([[-0.00235395 , 0.99988123 ,-0.01523124],
                              [-0.99998543, -0.00227965, 0.0048937],
                              [0.00485839, 0.01524254, 0.99987202]])
  translation_vector = np.array([-0.09321419, 0.03625434, 0.02420657])
  
  def convert(x ,y ,z):
      """
      我们需要将旋转向量和平移向量转换为齐次变换矩阵，然后使用深度相机识别到的物体坐标（x, y, z）和
      标定好的 相机到基座的 齐次变换矩阵 来计算物体相对于机械臂基座的位姿（x, y, z）
  
      """
  
  
  
      # 深度相机识别物体返回的坐标
      obj_camera_coordinates = np.array([x, y, z])
  
  
      # 将旋转矩阵和平移向量转换为齐次变换矩阵
      T_camera_to_base_effector = np.eye(4)
      T_camera_to_base_effector[:3, :3] = rotation_matrix
      T_camera_to_base_effector[:3, 3] = translation_vector
  
  
  
      # 计算物体相对于机械臂基座的位姿
      obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1])  # 将物体坐标转换为齐次坐标
  
      obj_base_effector_coordinates_homo = T_camera_to_base_effector.dot(obj_camera_coordinates_homo)
      obj_base_coordinates = obj_base_effector_coordinates_homo[:3]  # 从齐次坐标中提取物体的x, y, z坐标
  
  
      # 组合结果
  
      return list(obj_base_coordinates)
  
  
  
  ```

  

- 物体在相机坐标系是一个位姿

  ```python
  
  import numpy as np
  from scipy.spatial.transform import Rotation as R
  
  
  # 相机坐标系到机械臂基坐标系的旋转矩阵和平移向量（手眼标定得到）
  rotation_matrix = np.array([[-0.00235395 , 0.99988123 ,-0.01523124],
                              [-0.99998543, -0.00227965, 0.0048937],
                              [0.00485839, 0.01524254, 0.99987202]])
  translation_vector = np.array([-0.09321419, 0.03625434, 0.02420657])
  
  def decompose_transform(matrix):
      """
      将矩阵转化为位姿
      """
  
      translation = matrix[:3, 3]
      rotation = matrix[:3, :3]
  
      # Convert rotation matrix to euler angles (rx, ry, rz)
      sy = np.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
      singular = sy < 1e-6
  
      if not singular:
          rx = np.arctan2(rotation[2, 1], rotation[2, 2])
          ry = np.arctan2(-rotation[2, 0], sy)
          rz = np.arctan2(rotation[1, 0], rotation[0, 0])
      else:
          rx = np.arctan2(-rotation[1, 2], rotation[1, 1])
          ry = np.arctan2(-rotation[2, 0], sy)
          rz = 0
  
      return translation, rx, ry, rz
  
  
  
  def convert(x ,y ,z,rx,ry,rz):
      """
      我们需要将旋转向量和平移向量转换为齐次变换矩阵，然后使用深度相机识别到的物体位姿（x, y, z,rx,ry,rz）和
      标定好的 相机到基座的 齐次变换矩阵 来计算物体相对于机械臂基座的位姿（x, y, z,rx,ry,rz）
  
      """
  
  
  
      # 深度相机识别物体返回的坐标
      obj_camera_coordinates = np.array([x, y, z,rx,ry,rz])
  
  
      # 将旋转矩阵和平移向量转换为齐次变换矩阵
      T_camera_to_base_effector = np.eye(4)
      T_camera_to_base_effector[:3, :3] = rotation_matrix
      T_camera_to_base_effector[:3, 3] = translation_vector
  
  
  
      # 计算物体相对于机械臂基座的位姿
      # 物体相对于相机的位姿转换为齐次变换矩阵
      position2 = obj_camera_coordinates[:3]
      orientation2 = R.from_euler('xyz', obj_camera_coordinates[3:], degrees=False).as_matrix()
  
      T_object_to_camera_effector = np.eye(4)
      T_object_to_camera_effector[:3, :3] = orientation2
      T_object_to_camera_effector[:3, 3] = position2
  
      obj_base_effector = T_camera_to_base_effector.dot(T_object_to_camera_effector)
  
  	result = decompose_transform(obj_base_effector)
  
      # 组合结果
  
      return result
  
  ```

​		代码中rotation_matrix和translation_vector变量分别是眼在手外**手眼标定**得到的**旋转矩阵**和**平移向量**

### 
