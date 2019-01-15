import pybullet as p
import time
import pybullet_data
from hamstir_gym.modder import Modder
from tqdm import tqdm

mod = Modder()
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# planeId = p.loadURDF("plane.urdf")
room=p.loadURDF("./hamstir_gym/data/room12x12.urdf")
mod.load(room)
# mod.randomize()
# 
pbar = tqdm()
frames = 0
try:
    while True:
        p.stepSimulation()
        p.getCameraImage(128,128,renderer=p.ER_BULLET_HARDWARE_OPENGL)
        mod.randomize()
        # time.sleep(1/20.)
        frames += 1
        pbar.update(frames) 
except:
    print('bye')