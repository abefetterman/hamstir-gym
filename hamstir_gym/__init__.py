from gym.envs.registration import register

register(
    id='hamstir-room-empty-v0',
    entry_point='hamstir_gym.envs:HamstirRoomEmptyEnv',
)