import pybullet as p
from pyquaternion import Quaternion
import os.path as osp
import numpy as np

DATA_DIR = osp.join(osp.abspath(osp.dirname(__file__)),'data/')

def getWallDistance(wall_obj,ref_obj, maxDistance = 1.0):
    closest = p.getClosestPoints(wall_obj,ref_obj,maxDistance)
    distance = maxDistance
    for point in closest:
        if point[3] != -1:
            # [3] is linkIndexA, ignore the floor
            distance = min(distance, point[8])
    return distance

def find_links(car):
    num_joints = p.getNumJoints(car)
    camera_link_id = -1
    right_wheel_id = -1
    left_wheel_id = -1
    for joint in range(num_joints):
        joint_name = p.getJointInfo(car, joint)[12]
        if joint_name == b'camera_link':
            camera_link_id = joint
        if joint_name == b'right_wheel_link':
            right_wheel_id = joint
        if joint_name == b'left_wheel_link':
            left_wheel_id = joint
    return camera_link_id, left_wheel_id, right_wheel_id
    
def get_camera_view(car, camera_link_id, cameraFocusVec=(4,0,0), cameraUpVec=(0,0,1)):
    camPos, camOrient = p.getLinkState(car, camera_link_id)[4:6]
    camQ = Quaternion(camOrient[-1:]+camOrient[:3])
    focusPos = [a+b for a,b in zip(camPos, camQ.rotate(cameraFocusVec))]
    return p.computeViewMatrix(camPos, focusPos, camQ.rotate(cameraUpVec))

