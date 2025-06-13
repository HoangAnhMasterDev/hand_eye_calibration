# coding=utf-8

# This script is used to collect data from a RealSense L515 camera and a robot arm.
# It captures images and the current pose of the robot arm when the 's' key is pressed.


import json
import logging,os
import socket
import time
import sys
import numpy as np
import cv2
import pyrealsense2 as rs


from libs.log_setting import CommonLog
from libs.auxiliary import create_folder_with_date, get_ip, popup_message

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


rtde_receive = None

cam0_origin_path = create_folder_with_date() # A pre-created directory for storing photo files


logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

def callback(frame):

    scaling_factor = 2.0
    global count

    cv_img = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    cv2.imshow("Capture_Video", cv_img)  # Window display, with the title name Capture_Video

    k = cv2.waitKey(30) & 0xFF  #  Each frame is delayed by 1 ms; the delay must not be 0, otherwise the result will be a static frame

    if k == ord('s'):  # If the 's' key is pressed, print the current frame

        state, pose = get_ur10_tcp_pose()
        logger_.info( f'Get status: {"Success" if state else "Failure"}, {f"Current pose is {pose}" if state else None}')

        #saving the pose
        if state:

            filename = os.path.join(cam0_origin_path,"poses.txt")

            with open(filename, 'a+') as f:
                # Join the elements in the list into a single line separated by spaces
                pose_ = [str(i) for i in pose]
                new_line = f'{",".join(pose_)}\n'
                # Append the new line to the end of the file
                f.write(new_line)

            #saving the image
            image_path = os.path.join(cam0_origin_path,f"{str(count)}.jpg")
            cv2.imwrite(image_path , cv_img)
            logger_.info(f"===Collecting data for {count}th time！")

        count += 1

    else:
        pass

def get_ur10_tcp_pose():
    """
    Get the current TCP pose of the UR10 using RTDE.
    Returns a 6-element list: [x, y, z, rx, ry, rz]
    Units: meters and radians
    """
    try:
        pose = rtde_receive.getActualTCPPose()
        if pose is None or len(pose) != 6:
            return False, "Invalid pose received"
        return True, pose
    except Exception as e:
        logger_.error(f"Failed to get pose: {e}")
        return False, str(e)
    
def displayL515():

    # CHANGE TO 515 camera
    #STEP 1: Start the RealSense camera stream

    pipeline = rs.pipeline()
    config = rs.config()

    ###### what resolution and fps should be set for the camera????? bgr8 or what format?????
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

    # This try: start streaming
    try:
        pipeline.start(config)
    except Exception as e:
        logger_.error_(f"Camera connection：{e}")
        popup_message("Notice", "Camera connection failed")

        sys.exit(1)

    global count
    count = 1

    logger_.info(f"Starting hand-eye calibration program with L515 - V1.0.0")

    # This try: wait for frames and process them
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            callback(color_image)

    finally:

        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    robot_ip = get_ip()

    logger_.info(f'robot_ip:{robot_ip}')


    try:
        rtde_receive
        rtde_receive = RTDEReceiveInterface(robot_ip)
        logger_.info("Connected to UR10 via RTDE")
    except Exception as e:
        popup_message("Error", f"Failed to connect to UR10: {e}")
        sys.exit(1)
    

    displayL515()
