import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy import ndimage
import pybullet as p
from hamstir_gym.utils import *
from hamstir_gym.multiroom import MultiRoom
from hamstir_gym.camera import Camera

class HamstirRoomEmptyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render=False, step_ratio=25, dim=128, discrete=False, colors=3):
        
        self.dim = dim
        self.vel_mult = 10.0
        
        if discrete:
            self.action_space = spaces.Discrete(5)
            self.vel = 0.5
            self.actions = [[self.vel, self.vel],
                                [-self.vel, -self.vel],
                                [self.vel, -self.vel],
                                [-self.vel, self.vel],
                                [0, 0]]
        else:
            self.action_space = spaces.Box(-1,1,(2,),dtype=np.float32)
            self.actions = None
            
        self.colors = colors
        self.observation_space = spaces.Box(low=0, high=255, shape=(dim, dim, colors), dtype=np.uint8) # returns RGB
        
        self.physicsClientId = -1
        self.ownsPhysicsClient = False
        self.isRender = render
        self.connection_mode = p.GUI # if render else p.DIRECT
        self.room = None
        self.robot = None
        self.step_ratio = step_ratio # render timesteps / step(); render tstep = 1/240 sec
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL # or p.ER_TINY_RENDERER
        self.maxForce = 10
        self.maxSteps = 250
        self.multiroom = MultiRoom()
        self.camera = Camera(dim, dim, colors)
        self.bufferWallDistance = 0.4
        self.videoFile = None
        self.seed()
        
        return
        
    def seed(self, seed=None):
        if seed:
            print('set seed %d' % seed)
        else:
            print('reset seed')
        self.np_random, seed = seeding.np_random(seed)
        if hasattr(self,'multiroom'):
            seed = self.multiroom.seed(seed)
        if hasattr(self,'camera'):
            seed = self.camera.seed(seed)
        return [seed]
        
    def _resetClient(self):
        if (self.physicsClientId>=0):
            p.resetSimulation()
            return
            
        self.ownsPhysicsClient = True

        self.physicsClientId = p.connect(self.connection_mode)
            
        if self.isRender:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        else:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
            
        if self.videoFile:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
            # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,1)
            # p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
            # p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.videoFile)
    
    def _loadSimulation(self):
        p.setGravity(0,0,-9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=0.01)
        
        p.setAdditionalSearchPath(DATA_DIR)
        
        self.multiroom.load(p) 
        
        cubeStartPos = [0,2,.2]
        cubeStartAngle = self.np_random.uniform()*2*np.math.pi - np.math.pi
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,cubeStartAngle])
        self.robot = p.loadURDF(DATA_DIR+"/car.urdf", cubeStartPos, cubeStartOrientation)
        
        self.camera_link_id, left_wheel_id, right_wheel_id = find_links(self.robot)
        self.wheel_ids = [left_wheel_id, right_wheel_id]
        
    def reset(self):
        self._resetClient()
        
        self._loadSimulation()
        
        self.multiroom.reset()
        
        cubeStartPos = [0,2,.2]
        # cubeStartPos = [0,0,.2]
        cubeStartAngle = self.np_random.uniform()*2*np.math.pi - np.math.pi
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,cubeStartAngle])
        p.resetBasePositionAndOrientation(self.robot, cubeStartPos, cubeStartOrientation)
        p.resetBaseVelocity(self.robot, [0,0,0], [0,0,0])
        
        self.camera.reset()
        
        self.ep_len, self.ep_reward = 0, 0.0
        
        return self._get_img()
        
        
    def step(self, action):
        startPosition,_ = p.getBasePositionAndOrientation(self.robot)
        
        wheel_speeds = self.actions[action] if self.actions else action
        for wheel, vel in zip(self.wheel_ids, wheel_speeds):
            # presumably targetVelocity is in radians/second, force is in N-m -- unverified
            vel = np.clip(vel, -1.0, 1.0)
            p.setJointMotorControl2(self.robot, wheel, p.VELOCITY_CONTROL, targetVelocity=vel*self.vel_mult, force=self.maxForce)

        for _ in range(self.step_ratio):
            p.stepSimulation()
        
        img_arr = self._get_img()
        
        endPosition,_ = p.getBasePositionAndOrientation(self.robot)
        travelDistance2 = sum([(x-y)*(x-y) for x,y in zip(startPosition,endPosition)])
        
        wallDistance = getWallDistance(self.multiroom.active_room(), self.robot, self.bufferWallDistance)
        
        done = False
        reward = 0
        if all([s>0 for s in wheel_speeds]):
            # add travel distance only if moving forward
            reward += 10 * np.sqrt(travelDistance2)
            
            
        if wallDistance < 0.05:
            done=True
            reward = -500
        elif wallDistance < self.bufferWallDistance:
            # penalty for being close to the wall
            reward -= 100 * ((self.bufferWallDistance-wallDistance) ** 2)
            # pass
            
        self.ep_len += 1
        self.ep_reward += reward
        if self.ep_len > self.maxSteps:
            done = True

        return img_arr, reward, done, {'episode': { 'r': self.ep_reward, 'l': self.ep_len }}
        
    def _get_img(self):
        return self.camera.getImage(self.robot, self.camera_link_id, self.renderer)

    def render(self, mode='human', close=False):

        return self._get_img()
        
    def logVideo(self,filename):
        self.videoFile = filename
        