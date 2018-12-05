import pybullet as p
import numpy as np

from hamstir_gym.utils import DATA_DIR, DATA_ROOMS
from hamstir_gym.modder import Modder

class MultiRoom:
    def __init__(self, dir=DATA_DIR, rooms=DATA_ROOMS):
        self.data_dir = dir
        self.data_rooms = rooms
        self.n_rooms = len(rooms)
        
    def load(self, physicsClient):
        self._p = physicsClient
        self.rooms = []
        self.modders = []
        for room_path in self.data_rooms:
            startPos = [0,0,-5]
            room = self._p.loadURDF(self.data_dir+room_path, startPos, useFixedBase=1)
            self.rooms.append(room)
            modder = Modder()
            modder.load(room)
            modder.hide()
            self.modders.append(modder)
        
        self.active = None
        
        # i = 0
        # self.active = i
        # startPos = [0,0,0]
        # startOrientation = p.getQuaternionFromEuler([0,0,0])
        # self._p.resetBasePositionAndOrientation(self.rooms[i], startPos, startOrientation)
        # self.modders[i].randomize()
        # self.modders[i].show()
        
        
    def reset_room(self):
        if self.active:
            i = self.active
            startPos = [0,0,-5]
            startOrientation = p.getQuaternionFromEuler([0,0,0])
            self._p.resetBasePositionAndOrientation(self.rooms[i], startPos, startOrientation)
            self.modders[i].hide()
        
        i = np.random.randint(self.n_rooms)
        startPos = [0,0,0]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self._p.resetBasePositionAndOrientation(self.rooms[i], startPos, startOrientation)
        self.modders[i].randomize()
        self.modders[i].show()
        self.active = i
        
        # self.modders[0].randomize()
        
    def active_room(self):
        return self.rooms[self.active]