import pybullet as p
from pyquaternion import Quaternion
import os.path as osp
import numpy as np
from scipy import ndimage
from gym.utils import seeding

class Camera(object):
    def __init__(self, h, w, d=3):
        self.h, self.w, self.d = h, w, d
        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed
        
    def reset(self):
        self.lightXYZ = (self.np_random.uniform(size=3)*20 - 10).tolist()
        self.lightXYZ[2] = 10
        self.lightRGB = (self.np_random.uniform(size=3) * .5 + 0.5).tolist()
        self.lightCoeff = self.np_random.uniform(size=3).tolist()
        self.camShift = (self.np_random.uniform() - 0.25)*0.05
        self.camFocus = 0.15 # (self.np_random.uniform() - 0.05)*1 + .1
        self.upVec = (0,0,1)
        fov = (self.np_random.uniform() - 0.5)*20 + 40
        self.cameraProjection = p.computeProjectionMatrixFOV(fov=fov, aspect=1.0, nearVal=0.05, farVal=20.0)
        
    def getView(self, robot, camera_link_id):
        camPos, camOrient = p.getLinkState(robot, camera_link_id, computeForwardKinematics=True)[4:6]
        camPos = (camPos[0],camPos[1],camPos[2]+self.camShift)
        camQ = Quaternion(camOrient[-1:]+camOrient[:3])
        focusPos = [a+b for a,b in zip(camPos, camQ.rotate((self.camFocus,0,0)))]
        return p.computeViewMatrix(camPos, focusPos, camQ.rotate(self.upVec))
        
    def getImage(self, robot, camera_link_id, renderer=p.ER_BULLET_HARDWARE_OPENGL):
        cameraView = self.getView(robot, camera_link_id)
        img_params = p.getCameraImage(self.w, self.h, cameraView, self.cameraProjection, \
                    lightDirection = self.lightXYZ, lightColor = self.lightRGB, \
                    lightAmbientCoeff = self.lightCoeff[0], lightDiffuseCoeff = self.lightCoeff[1], \
                    lightSpecularCoeff = self.lightCoeff[2], renderer=renderer)
        img = img_params[2][...,:self.d]
        img = ndimage.gaussian_filter(img, sigma=(0.5,0.5,0))
        return img
        