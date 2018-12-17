import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
from hamstir_gym.utils import *
from hamstir_gym.multiroom import MultiRoom

class HamstirRoomEmptyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render=False, step_ratio=25, discrete=False):
        
        self.camera_height, self.camera_width = 80,80
        
        if discrete:
            self.action_space = spaces.Discrete(5)
            self.actions = [(0,0),(10,10),(-5,-5),(5,-5),(-5,5)]
        else:
            self.action_space = spaces.Box(-10,10,(2,),dtype=np.float32)
            self.actions = None
            
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.camera_height, self.camera_width, 4), dtype=np.uint8) # returns RGBA
        
        self.physicsClientId = -1
        self.ownsPhysicsClient = False
        self.isRender = render
        self.room = None
        self.robot = None
        self.step_ratio = step_ratio # render timesteps / step(); render tstep = 1/240 sec
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL # or p.ER_TINY_RENDERER
        self.maxForce = 10
        self.maxSteps = 120
        
        return
    
    def _resetClient(self):
        if (self.physicsClientId>=0):
            # p.resetSimulation()
            return
            
        self.ownsPhysicsClient = True

        # if self.isRender:
        #     p = bullet_client.BulletClient(connection_mode=p.GUI)
        # else:
        #     p = bullet_client.BulletClient()
        # p = p

        self.physicsClientId = p.connect(p.GUI)
        self.multiroom = MultiRoom()
        self.cameraProjection = p.computeProjectionMatrixFOV(fov=90.0, aspect=1.0, nearVal=0.1, farVal=10.0)
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
    
        p.setGravity(0,0,-9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=0.01)
        
        p.setAdditionalSearchPath(DATA_DIR)
        
        self.multiroom.load(p) 
        
        cubeStartPos = [0,2,.2]
        cubeStartAngle = np.random.uniform()*2*np.math.pi - np.math.pi
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,cubeStartAngle])
        self.robot = p.loadURDF(DATA_DIR+"/car.urdf", cubeStartPos, cubeStartOrientation)
        
        self.camera_link_id, left_wheel_id, right_wheel_id = find_links(self.robot)
        self.wheel_ids = [left_wheel_id, right_wheel_id]
        
    
    def reset(self):
        self._resetClient()
        
        self.multiroom.reset()
        
        cubeStartPos = [0,2,.2]
        cubeStartAngle = np.random.uniform()*2*np.math.pi - np.math.pi
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,cubeStartAngle])
        p.resetBasePositionAndOrientation(self.robot, cubeStartPos, cubeStartOrientation)
        p.resetBaseVelocity(self.robot, [0,0,0], [0,0,0])
        
        self.ep_len, self.ep_reward = 0, 0.0
        
        return self._get_img()
        
        
    def step(self, action):
        startPosition,_ = p.getBasePositionAndOrientation(self.robot)
        
        wheel_speeds = self.actions[action] if self.actions else action
        for wheel, vel in zip(self.wheel_ids, wheel_speeds):
            # presumably targetVelocity is in radians/second, force is in N-m -- unverified
            p.setJointMotorControl2(self.robot, wheel, p.VELOCITY_CONTROL, targetVelocity=vel, force=self.maxForce)

        for _ in range(self.step_ratio):
            p.stepSimulation()
        
        img_arr = self._get_img()
        
        endPosition,_ = p.getBasePositionAndOrientation(self.robot)
        travelDistance2 = sum([(x-y)*(x-y) for x,y in zip(startPosition,endPosition)])
        
        wallDistance = getWallDistance(self.multiroom.active_room(), self.robot)
        
        done = False
        reward = 0
        if all([s>0 for s in wheel_speeds]):
            # add travel distance only if moving forward
            reward += 10 * np.sqrt(travelDistance2)
            
        if wallDistance < 0.01:
            done=True
            reward = -500
        elif wallDistance < .1:
            # penalty for being close to the wall
            reward -= 1e3 * ((.1-wallDistance) ** 2)
            # pass
            
        self.ep_len += 1
        self.ep_reward += reward
        if self.ep_len > self.maxSteps:
            done = True

        return img_arr, reward, done, {'episode': { 'r': self.ep_reward, 'l': self.ep_len }}
        
    def _get_img(self):
        cameraView = get_camera_view(self.robot, self.camera_link_id)
        img_params = p.getCameraImage(self.camera_width, self.camera_height, cameraView, self.cameraProjection, renderer=self.renderer)
        return img_params[2]

    def render(self, mode='human', close=False):

        return self._get_img()