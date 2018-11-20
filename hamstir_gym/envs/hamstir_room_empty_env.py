import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet
from pybullet_utils import bullet_client

class HamstirRoomEmptyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render=False):
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0
        self.isRender = render
        self.room = None
        self.robot = None
        
        return
    def step(self, action):
        return
    def reset(self):
        if (self.physicsClientId<0):
            self.ownsPhysicsClient = True

			if self.isRender:
      			self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
			else:
				self._p = bullet_client.BulletClient()
	
			self.physicsClientId = self._p._client
			self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
        
            self._p.setGravity(0,0,-10)
            
            self.room = self._p.loadURDF("urdf/room.urdf", useFixedBase=1)
            cubeStartPos = [0,2,.05]
            cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
            self.robot = self._p.loadURDF("urdf/car.urdf", cubeStartPos, cubeStartOrientation)
        
        return
    def render(self, mode='human', close=False):
        return