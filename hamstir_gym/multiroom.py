import pybullet as p
import numpy as np
from gym.utils import seeding

from hamstir_gym.utils import DATA_DIR, DATA_ROOMS
from hamstir_gym.modder import Modder

class MultiRoom:
    def __init__(self, dir=DATA_DIR, rooms=DATA_ROOMS):
        self.data_dir = dir
        self.data_rooms = rooms
        self.n_rooms = len(rooms)
        self.modder = Modder()
        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        if hasattr(self,'modder'):
            seed = self.modder.seed(seed)
        return seed
        
    def load(self, physicsClient):
        self._p = physicsClient
        i = self.np_random.randint(self.n_rooms)
        startPos = [0,0,0]
        self.room = self._p.loadURDF(self.data_dir+self.data_rooms[i], startPos, useFixedBase=1)
        # self.room = self._p.loadSDF("/Users/abe/clones/bullet3/data/kitchens/1.sdf")
        
        self.modder.load(self.room)
        
    def reset(self):
        self.modder.randomize()
        return
        
    def active_room(self):
        return self.room