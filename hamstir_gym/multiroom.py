import pybullet as p
import numpy as np

from hamstir_gym.utils import DATA_DIR, DATA_ROOMS
from hamstir_gym.modder import Modder

class MultiRoom:
    def __init__(self, dir=DATA_DIR, rooms=DATA_ROOMS):
        self.data_dir = dir
        self.data_rooms = rooms
        self.n_rooms = len(rooms)
        self.modder = Modder()
        
    def load(self, physicsClient):
        self._p = physicsClient
        i = np.random.randint(self.n_rooms)
        startPos = [0,0,0]
        self.room = self._p.loadURDF(self.data_dir+self.data_rooms[i], startPos, useFixedBase=1)
        
        self.modder.load(self.room)
        
    def reset(self):
        self.modder.randomize()
        
    def active_room(self):
        return self.room