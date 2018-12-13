import importlib
from gym.envs.registration import register

register(
    id='hamstir-room-empty-v0',
    entry_point='hamstir_gym.envs:HamstirRoomEmptyEnv',
)

if (importlib.find_loader('gibson') != None):
    register(
        id='hamstir-gibson-v0',
        entry_point='hamstir_gym.envs:HamstirGibsonEnv',
    )