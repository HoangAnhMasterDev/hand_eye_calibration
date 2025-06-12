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

    # Sub-pixel corner detection criteria
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # Prepare the 3D coordinates of the calibration board points
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


The above code iterates through the collected calibration board images, detects the chessboard corners one by one, and stores them in an array.

#### 2.Camera Calibration

**compute_in_hand.py | compute_to_hand.py**

```python


# Calibration to obtain the pose of the pattern in the camera coordinate system
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

```



- 使用OpenCV的 `calibrateCamera` The function performs camera calibration to compute the camera’s intrinsic parameters and distortion coefficients。

- `rvecs` 和 `tvecs` It also provides, for each image, a rotation vector and a translation vector, representing the**pose of the calibration board in the camera coordinate system**。

  

#### 3.Processing robot arm pose data

**compute_in_hand.py**

```python
poses_main(file_path)
```

Convert the end-effector pose data into the rotation matrix and translation vector of the robot end-effector coordinate system relative to the base coordinate system.

**compute_to_hand.py**

```python
poses2_main(file_path)
```

Convert the end-effector pose data into the rotation matrix and translation vector of the base coordinate system relative to the robot end-effector coordinate system.


####  4.Hand-eye calibration calculation

```python
R, t = cv2.calibrateHandEye(R_tool, t_tool, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)

return R,t

```

- 使用OpenCV的 `calibrateHandEye` The function performs hand-eye calibration.

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

    

- 采用了 `CALIB_HAND_EYE_TSAI` Method: This is a commonly used hand-eye calibration algorithm



### `save_poses.py` Key code explanation

#### 1. **Define a function to convert Euler angles to a rotation matrix**

```python
def euler_angles_to_rotation_matrix(rx, ry, rz):
    # Compute the rotation matrix
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

- A function is defined to convert Euler angles (rotations around the X, Y, and Z axes) into a rotation matrix.。
- The rotations are applied in Z-Y-X order, and the rotation matrices for each axis are multiplied together to obtain the final rotation matrix R.

#### 2. **Convert pose to homogeneous transformation matrix**

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

- Convert the pose (position and orientation) into a 4×4 homogeneous transformation matrix for matrix operations.
- Position `(x, y, z)`  is used as the translation vector, and orientation `(rx, ry, rz)` as Euler rotation angles。
- The resulting homogeneous transformation matrix H describes the rotation and translation of the robot end-effector relative to the base.

#### 3. **Save multiple matrices to a CSV file**

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

- Concatenate multiple matrices horizontally into a single large matrix and save it to a CSV file.。

- Each matrix occupies a fixed number of columns, allowing you to split them into individual matrices at fixed intervals when reading.。

  

## Calibration Process

### 1.Environment Requirements

#### Basic Environment Setup

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

Run the following command in your Python environment to install the packages required for the hand-eye calibration program:：

```cmd
pip install -r requirements.txt
```



#### HARDWARE

- ARM：RM75 RM65  RM63 GEN72 
- CAM: Intel RealSense Depth Camera D435

- Dedicated Camera Data Cable: For stable and high-quality data transmission between the camera and computer.

- Ethernet Cable: For connecting the robot or camera to the computer or network.

- Calibration Board (1 or 2 units): Can be a rigid board with a printed pattern (checkerboard or circle grid).

  1. Printed Paper Calibration Board

     ![标定板图片_00(1)](picture/标定板图片_00(1).png)

  2. Purchase calibration board checkerboard if wanted







### 2.Calibration process

#### Config 

Set the calibration board parameters in the configuration file（config.yaml

![image-20250403064543828](picture/image-20250403064543828.png)

​		
The configuration parameters in the file are as follows:

​    xx: Number of horizontal corners on the calibration board (number of squares along the long edge minus 1), default is 11.
For example, if there are 12 squares along the long edge, there are 11 inner corners.

​		YY: Number of vertical corners on the calibration board (number of squares along the short edge minus 1), default is 8.
For example, if there are 9 squares along the short edge, there are 8 inner corners.

​		L :  Size of a single square on the calibration board (unit: meters), default is 0.03.
This means each square is 3 cm wide in the real world.

​        ![](picture/image-20241016181226851.png)



#### DATA COLLECTION

#####Connect the device

(1) Connect the D435 camera to the computer using the camera data cable.

(2) Connect the robot arm to the computer using an Ethernet cable.

If the robot arm's IP address is 192.168.1.18, then the computer's IP address should be set to the same subnet, i.e., the 192.168.1.x range.

![image-20241018143659263](picture/image-20241018143659263.png)

​If the robot arm's IP address is 192.168.10.18, then the computer’s IP address should be set to the 192.168.10.x subnet (as described above).                                     

(3) The calibration board is placed flat on a surface, the camera is fixed to the robot arm’s end-effector, and the camera is aimed at the calibration board.

​		During the calibration process, the calibration board is fixed in place within the robot arm’s workspace. This fixed position must ensure that the camera mounted on the robot’s end-effector can observe the board from different viewpoints. The exact position of the calibration board is not important because its pose relative to the robot base does not need to be known. However, the calibration board must remain stationary and must not be moved throughout the calibration.

(4) Run `collect_data.py`，and a popup window will appear

(5) Manually move the robot arm’s end-effector to position the calibration board clearly and fully within the camera’s field of view, then place the cursor over the popup window.

​		**Note：**

- Ensure that the calibration board and the camera lens in the camera’s field of view form a certain angle.

  ![image-20241101112801145](picture/image-20241101112801145.png)

  The following camera pose is incorrect:
  ![image-20241101112816914](picture/image-20241101112816914.png)

​				

(6) Press the “s” key on the keyboard to capture the data.

(7) Move the robot arm 15–20 times, repeating steps (5) and (6) to capture about 15–20 images of the calibration board from different robot poses.

​	**NOTE：**

​			Move the robot arm’s end-effector rotation axes, making sure each rotation angle is as large as possible (greater than 30°) each time.

​			Make sure there are sufficient rotation angle variations along all three axes (X, Y, and Z).

​			(You can first rotate the robot arm’s end-effector around the Z-axis multiple times to capture several images, then rotate it around the X-axis for additional views.)

​			![WPS拼图1](picture/WPS拼图1.png)

##### Calculate the calibration results

​		Run the script `compute_in_hand.py`，to obtain the calib result

​        Obtain the rotation matrix and translation vector of the camera coordinate system relative to the robot arm end-effector coordinate system.

#### Eye-to-hand calibration (Eye outside hand)

##### Data collection

(1) Connect the camera cable between the computer and the D435 camera, and connect the Ethernet cable between the computer and the robot arm.

(2) Set the computer’s IP address and the robot arm’s IP address to the same subnet.

​	If the robot arm’s IP address is 192.168.1.18, then set the computer’s IP address to the 192.168.1.x subnet.

​  If the robot arm’s IP address is 192.168.10.18, then set the computer’s IP address to the 192.168.10.x subnet.

(3) Fix the calibration board (a smaller printed paper board for easy mounting) to the robot arm’s end-effector, keep the camera stationary, and move the robot arm’s end-effector so that the calibration board appears within the camera’s field of view.

​       During the calibration process, the calibration board is mounted on the robot arm’s end-effector and moves together with the arm. It can be directly fixed to the tool flange or attached using a fixture. The exact mounting position is not important because the relative pose between the calibration board and the end-effector does not need to be known. What matters is that the calibration board does not move relative to the tool flange or fixture during motion — it must be firmly fixed or tightly clamped. It is recommended to use a mounting bracket made of rigid material to ensure stability.

(4) Run the script collect_data.py, and a popup window will appear.

(5) Manually move the robot arm’s end-effector to position the calibration board clearly and fully within the camera’s field of view, then place the cursor over the popup window.

​		**NOTE：**

​				Make sure the calibration board and the camera lens in the camera’s field of view form a certain angle。

(6) Press the “s” key on the keyboard to capture the data.

(7)Move the robot arm 15–20 times, repeating steps (5) and (6) to capture about 15–20 images of the calibration board from different robot poses.

​			Rotate the robot arm’s end-effector rotation axes, making each rotation as large as possible (greater than 30°) each time.

​			Ensure there is sufficient rotation variation along all three axes (X, Y, and Z).



##### Calculate Calibration Results

Run the script `compute_to_hand.py`，to obtain the calibration results.

Obtain the rotation matrix and translation vector of the camera coordinate system relative to the robot arm base coordinate system.

The translation vector is in units of meters.

#### Error Range

Affected by the quality of the captured images, the discrepancy between the translation vector in the calibration results and the actual value is within 1 cm.。

## Possible Issues During Calibration Process

### Issue1

When calculating the collected data,

that is, when running the following scripts：

```cmd
python compute_in_hand.py
```

或

```python
python compute_to_hand.py
```

the following problem may occur:

问题描述：[ERROR:0@1.418] global calibration_handeye.cpp:335 calibrateHandEyeTsai Hand-eye calibration failed! Not enough informative motions--include larger rotations.



**Cause**：Insufficient rotational movement in the collected images, especially lacking sufficiently large rotational motions.

Hand-eye calibration requires the robot to perform a series of movements in space to accurately determine the camera’s pose relative to the robot’s end-effector. If the robot’s motion data lacks sufficient rotational variation, particularly significant rotations along each axis, the calibration algorithm cannot accurately compute the hand-eye relationship.


**Solution：**

1. **Increase rotational movement：**

- During data collection, make the robot perform larger rotational motions. Ensure sufficient rotation angle variation along all three axes (X, Y, Z).
- For example, rotation angles should ideally exceed 30 degrees to provide rich motion information.

2. **Diversify motion poses: **

- During data capture, ensure the robot’s end-effector executes diverse poses, including translation and rotation.
- Avoid moving only within a small range or along a single axis.


3. **Increase data collection samples：**

- Collect more sample data; generally, at least 10 different poses are needed.
- More data improves calibration stability and accuracy.

## How to Use Hand-Eye Calibration Results

How does the robot arm pick up objects?

### Eye-in-Hand

#### Theory

Start from a robot arm without involving the camera. Its two main coordinate systems are:

​	1.Robot arm base coordinate system

   2.End-effector coordinate system

![../../../_images/hand-eye-robot-ee-robot-base-coordinate-systems.png](picture/hand-eye-robot-ee-robot-base-coordinate-systems-17301939637152.png)






To pick up an object, the robot arm needs to know the pose of the object relative to the robot arm base coordinate system. Using this information along with the robot’s geometric knowledge, the joint angles for moving the end-effector/gripper toward the object can be calculated.

![../../../_images/hand-eye-robot-robot-to-object.png](picture/hand-eye-robot-robot-to-object.png)

The pose of the object relative to the camera coordinate system can be obtained through model recognition. To allow the robot arm to grasp the object, the object’s pose must be transformed from the camera coordinate system to the robot arm base coordinate system.

![../../../_images/hand-eye-robot-ee-robot-base-coordinate-systems-with-camera.png](picture/hand-eye-robot-ee-robot-base-coordinate-systems-with-camera.png)

In this case, the coordinate transformation is done indirectly:

$$ H^{ROB}{OBJ} = H^{ROB}{EE} \cdot H^{EE}{CAM} \cdot H^{CAM}{OBJ} $$

The pose of the end-effector relative to the robot base

$$H^{ROB}_{EE}$$

is known and can be obtained via the robot arm API; the pose of the camera relative to the end-effector

$$H^{EE}_{CAM}$$

is obtained through hand-eye calibration.

![img](picture/hand-eye-eye-in-hand-all-poses.png)



If the object is represented as a 3D point or a pose in the camera coordinate system, the following explains how to mathematically transform the object’s position as a 3D point or pose from the camera coordinate system to the robot arm base coordinate system.

![../../../_images/hand-eye-eye-in-hand-all-poses.png](picture/hand-eye-eye-in-hand-all-poses-17302010389227.png)

The robot arm’s poses are expressed as homogeneous transformation matrices.

The following equation describes how to transform a single 3D point from the camera coordinate system to the robot arm base coordinate system:

$$ p^{ROB} = H^{ROB}{EE} \cdot H^{EE}{CAM} \cdot p^{CAM} $$

$$ \begin{bmatrix} x^r \\\ y^r \\\ z^r \\\ 1 \end{bmatrix} = \begin{bmatrix} R_e^r & t_e^r \\\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} R_c^e & t_c^e \\\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} x^c \\\ y^c \\\ z^c \\\ 1 \end{bmatrix} $$

If transforming the object’s pose from the camera coordinate system to the robot arm base coordinate system:

$$ H^{ROB}{OBJ} = H^{ROB}{EE} \cdot H^{EE} {CAM} \cdot H^{CAM} {OBJ} $$

$$ \begin{bmatrix} R_o^r & t_o^r \\\ 0 & 1 \end{bmatrix} = \begin{bmatrix} R_e^r & t_e^r \\\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} R_c^e & t_c^e \\\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} R_o^c & t_o^c \\\ 0 & 1 \end{bmatrix} $$


​	The resulting pose corresponds to the pose the robot’s current tool coordinate frame center should reach to perform the pick operation. (The robot poses collected during calibration are also the poses of the current tool coordinate frame relative to the robot base coordinate system.)



#### Code



- The object is a 3D point (x, y, z) in the camera coordinate system

  ```python
  
  import numpy as np
  from scipy.spatial.transform import Rotation as R
  
  
  # Rotation matrix and translation vector from camera coordinate system to robot end-effector coordinate system (obtained by hand-eye calibration)
  rotation_matrix = np.array([[-0.00235395 , 0.99988123 ,-0.01523124],
                              [-0.99998543, -0.00227965, 0.0048937],
                              [0.00485839, 0.01524254, 0.99987202]])
  translation_vector = np.array([-0.09321419, 0.03625434, 0.02420657])
  
  
  def convert(x ,y ,z ,x1 ,y1 ,z1 ,rx ,ry ,rz):
      """
          We need to convert the rotation vector and translation vector obtained by hand-eye calibration into a homogeneous transformation matrix, then use the object coordinates (x, y, z) recognized by the depth camera and the robot end-effector pose (x1,y1,z1,rx,ry,rz) to calculate the object's pose relative to the robot base (x, y, z).

  
      """
  
  
  
     # Coordinates of the object recognized by the depth camera

      obj_camera_coordinates = np.array([x, y, z])
  
      # Robot end-effector pose, in radians
      end_effector_pose = np.array([x1, y1, z1,
                                    rx, ry, rz])
  
      # Convert rotation matrix and translation vector into homogeneous transformation matrix
      T_camera_to_end_effector = np.eye(4)
      T_camera_to_end_effector[:3, :3] = rotation_matrix
      T_camera_to_end_effector[:3, 3] = translation_vector
  
      # Convert robot end-effector pose to homogeneous transformation matrix
      position = end_effector_pose[:3]
      orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()
  
      T_base_to_end_effector = np.eye(4)
      T_base_to_end_effector[:3, :3] = orientation
      T_base_to_end_effector[:3, 3] = position
  
      # Calculate the object's pose relative to the robot base
      obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1])  # 将物体坐标转换为齐次坐标
  
      obj_end_effector_coordinates_homo = T_camera_to_end_effector.dot(obj_camera_coordinates_homo)
  
      obj_base_coordinates_homo = T_base_to_end_effector.dot(obj_end_effector_coordinates_homo)
  
      obj_base_coordinates = list(obj_base_coordinates_homo[:3])  # 从齐次坐标中提取物体的x, y, z坐标
  
  
      return obj_base_coordinates
  
  
  ```

  

-The object is a pose in the camera coordinate system

  ```python
  
  import numpy as np
  from scipy.spatial.transform import Rotation as R
  
  
  # Rotation matrix and translation vector from camera coordinate system to robot end-effector coordinate system (obtained by hand-eye calibration)

  rotation_matrix = np.array([[-0.00235395 , 0.99988123 ,-0.01523124],
                              [-0.99998543, -0.00227965, 0.0048937],
                              [0.00485839, 0.01524254, 0.99987202]])
  translation_vector = np.array([-0.09321419, 0.03625434, 0.02420657])
  
  
  def decompose_transform(matrix):
      """
      Convert matrix to pose
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
  
      We need to convert the rotation vector and translation vector obtained by calibration into a homogeneous transformation matrix, then use the robot end-effector pose (x, y, z,rx,ry,rz) and the object coordinates recognized by the depth camera (x1,y1,z1,rx1,ry1,rz1) to calculate the object's pose relative to the robot base
  
      """
  
  
  
      # Object coordinates recognized by the depth camera
      obj_camera_coordinates = np.array([x1, y1, z1,rx1,ry1,rz1])
  
      # Robot end-effector pose, in radians
      end_effector_pose = np.array([x, y, z,
                                    rx, ry, rz])
  
      # Convert rotation matrix and translation vector into homogeneous transformation matrix
      T_camera_to_end_effector = np.eye(4)
      T_camera_to_end_effector[:3, :3] = rotation_matrix
      T_camera_to_end_effector[:3, 3] = translation_vector
  
      # Convert robot end-effector pose to homogeneous transformation matrix
      position = end_effector_pose[:3]
      orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()
  
      T_end_to_base_effector = np.eye(4)
      T_end_to_base_effector[:3, :3] = orientation
      T_end_to_base_effector[:3, 3] = position
  
      # Calculate the object's pose relative to the robot base
  
  
      # Convert object's pose relative to camera into homogeneous transformation matrix
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
  
  

​		   he variables rotation_matrix and translation_vector in the code are the rotation matrix and translation vector obtained by hand-eye calibration with the camera on the robot's end-effector.

### Camera outside the hand

#### Theory

The camera can obtain the object's pose in the camera coordinate system through the model. The object pose relative to the robot is obtained by multiplying the camera pose relative to the robot base coordinate system and the object's pose relative to the camera coordinate system:

$$ H^{ROB}{OBJ} = H^{ROB}{CAM} \cdot H^{CAM}{OBJ} $$

![../../../_images/hand-eye-eye-to-hand-all-poses.png](picture/hand-eye-eye-to-hand-all-poses.png)



If the object position is a 3D point or a pose, below describes mathematically how to convert the object's position as a 3D point or pose from the camera coordinate system to the robot base coordinate system.

The following equation describes how to convert a single 3D point from camera coordinate system to robot base coordinate system：

$$ p^{ROB} = H^{ROB}{CAM} \cdot p^{CAM} $$

$$ \begin{bmatrix} x^r \\\ y^r \\\ z^r \\\ 1 \end{bmatrix} = \begin{bmatrix} R_c^r & t_c^r \\\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} x^c \\\ y^c \\\ z^c \ 1 \end{bmatrix} $$

If you want to convert the object pose from camera coordinate system to robot base coordinate system:

$$ H^{ROB}{OBJ} = H^{ROB}{CAM} \cdot H^{CAM}{OBJ} $$

$$ \begin{bmatrix} R_o^r & t_o^r \\\ 0 & 1 \end{bmatrix} = \begin{bmatrix} R_c^r & t_c^r \\\ 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} R_o^c & t_o^c \\\ 0 & 1 \end{bmatrix} $$

#### Code

- Object is a 3D point (x, y, z) in camera coordinate system

  ```python
  
  import numpy as np
  
  
  
  # Rotation matrix and translation vector from camera coordinate system to robot base coordinate system (obtained by hand-eye calibration)

  rotation_matrix = np.array([[-0.00235395 , 0.99988123 ,-0.01523124],
                              [-0.99998543, -0.00227965, 0.0048937],
                              [0.00485839, 0.01524254, 0.99987202]])
  translation_vector = np.array([-0.09321419, 0.03625434, 0.02420657])
  
  def convert(x ,y ,z):
      """
      We need to convert the rotation vector and translation vector into a homogeneous transformation matrix, then use the object coordinates (x, y, z) recognized by the depth camera and the calibrated camera-to-base homogeneous transformation matrix to calculate the object's pose relative to the robot base (x, y, z)

  
      """
  
  
  
      # Coordinates of the object recognized by the depth camera
      obj_camera_coordinates = np.array([x, y, z])
  
  
      # Convert rotation matrix and translation vector into homogeneous transformation matrix
      T_camera_to_base_effector = np.eye(4)
      T_camera_to_base_effector[:3, :3] = rotation_matrix
      T_camera_to_base_effector[:3, 3] = translation_vector
  
  
  
      # Calculate the object's pose relative to the robot base
      obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1])  # 将物体坐标转换为齐次坐标
  
      obj_base_effector_coordinates_homo = T_camera_to_base_effector.dot(obj_camera_coordinates_homo)
      obj_base_coordinates = obj_base_effector_coordinates_homo[:3]  # 从齐次坐标中提取物体的x, y, z坐标
  
  
      # Combine result
  
      return list(obj_base_coordinates)
  
  
  
  ```

  

- Object is a pose in camera coordinate system

  ```python
  
  import numpy as np
  from scipy.spatial.transform import Rotation as R
  
  
  # Rotation matrix and translation vector from camera coordinate system to robot base coordinate system (obtained by hand-eye calibration)

  rotation_matrix = np.array([[-0.00235395 , 0.99988123 ,-0.01523124],
                              [-0.99998543, -0.00227965, 0.0048937],
                              [0.00485839, 0.01524254, 0.99987202]])
  translation_vector = np.array([-0.09321419, 0.03625434, 0.02420657])
  
  def decompose_transform(matrix):
      """
      Convert matrix to pose
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
      We need to convert the rotation vector and translation vector into a homogeneous transformation matrix, then use the object pose (x, y, z,rx,ry,rz) recognized by the depth camera and the calibrated camera-to-base homogeneous transformation matrix to calculate the object's pose relative to the robot base (x, y, z,rx,ry,rz)

  
      """
  
  
  
      # Coordinates of the object recognized by the depth camera
      obj_camera_coordinates = np.array([x, y, z,rx,ry,rz])
  
  
      # Convert rotation matrix and translation vector into homogeneous transformation matrix
      T_camera_to_base_effector = np.eye(4)
      T_camera_to_base_effector[:3, :3] = rotation_matrix
      T_camera_to_base_effector[:3, 3] = translation_vector
  
  
  
      # Calculate the object's pose relative to the robot base
      # Convert object's pose relative to camera into homogeneous transformation matrix
      position2 = obj_camera_coordinates[:3]
      orientation2 = R.from_euler('xyz', obj_camera_coordinates[3:], degrees=False).as_matrix()
  
      T_object_to_camera_effector = np.eye(4)
      T_object_to_camera_effector[:3, :3] = orientation2
      T_object_to_camera_effector[:3, 3] = position2
  
      obj_base_effector = T_camera_to_base_effector.dot(T_object_to_camera_effector)
  
  	result = decompose_transform(obj_base_effector)
  
      # Combine result
  
      return result
  
  ```

​		The variables rotation_matrix and translation_vector in the code are the rotation matrix and translation vector obtained by hand-eye calibration with the camera outside the robot hand.

### 
