import pybullet as p
import os.path as osp
import numpy as np

DATA_DIR = osp.join(osp.abspath(osp.dirname(__file__)),'data/')
DATA_ROOMS = [
    "/room6x6.urdf",
    "/room2x12.urdf",
    "/room12x12.urdf"
]

def getWallDistance(wall_obj,ref_obj, maxDistance = 0.4):
    if type(wall_obj) == int:
        wall_obj = [wall_obj]
    distance = maxDistance
    for obj in wall_obj:
        closest = p.getClosestPoints(obj,ref_obj,maxDistance)
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
        if joint_name == b'eyes':
            camera_link_id = joint
        if joint_name == b'right_wheel_link':
            right_wheel_id = joint
        if joint_name == b'left_wheel_link':
            left_wheel_id = joint
    return camera_link_id, left_wheel_id, right_wheel_id
