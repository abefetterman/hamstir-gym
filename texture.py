import pybullet as p
import time
import pybullet_data
from hamstir_gym.modder import Modder

mod = Modder()
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# planeId = p.loadURDF("plane.urdf")
room=p.loadURDF("./hamstir_gym/data/room.urdf")
mod.load(room)
mod.randomize()

try:
    while True:
        p.stepSimulation()
        time.sleep(1/20.)
except:
    print('bye')