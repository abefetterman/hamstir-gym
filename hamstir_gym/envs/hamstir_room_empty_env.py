import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet
from pybullet_utils import bullet_client
from hamstir_gym.utils import *

class HamstirRoomEmptyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render=True, step_ratio=10):
        self.physicsClientId = -1
        self.ownsPhysicsClient = False
        self.isRender = render
        self.room = None
        self.robot = None
        self.step_ratio = step_ratio
        self.renderer = pybullet.ER_BULLET_HARDWARE_OPENGL
        self.maxForce = 10
        
        return
    
    def _resetClient(self):
        if (self.physicsClientId>=0):
            return
            
        self.ownsPhysicsClient = True

        if self.isRender:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.physicsClientId = self._p._client
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
    
        self._p.setGravity(0,0,-10)
        
        self._p.setAdditionalSearchPath(DATA_DIR)
        
        self.room = self._p.loadURDF(DATA_DIR+"/room.urdf", useFixedBase=1)
        
        cubeStartPos = [0,2,.05]
        cubeStartOrientation = pybullet.getQuaternionFromEuler([0,0,0])
        self.robot = self._p.loadURDF(DATA_DIR+"/car.urdf", cubeStartPos, cubeStartOrientation)
        
        self.camera_link_id, left_wheel_id, right_wheel_id = find_links(self.robot)
        self.wheel_ids = [left_wheel_id, right_wheel_id]
        
        self.cameraProjection = pybullet.computeProjectionMatrixFOV(fov=90.0, aspect=1.0, nearVal=0.1, farVal=10.0)
        
    
    def reset(self):
        self._resetClient()
        
        randomizeColors(self.room)
        
        return
        
        
    def step(self, action):
        for wheel, vel in zip(self.wheel_ids, action):
            # presumably targetVelocity is in radians/second, force is in N-m -- unverified
            p.setJointMotorControl2(self.robot, wheel, p.VELOCITY_CONTROL, targetVelocity=vel, force=self.maxForce)

        for _ in range(self.step_ratio):
            self._p.stepSimulation()
        
        cameraView = get_camera_view(self.robot, self.camera_link_id)
        img_arr = self._p.getCameraImage(240, 240, cameraView, self.cameraProjection, renderer=self.renderer)

        return img_arr
        
    def render(self, mode='human', close=False):
        cameraView = get_camera_view(self.robot, self.camera_link_id)
        img_arr = self._p.getCameraImage(240, 240, cameraView, self.cameraProjection, renderer=self.renderer)

        return img_arr