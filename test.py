import gym
import hamstir_gym

env = gym.make('hamstir-room-empty-v0')

env.reset()
for _ in range(1000):
    env.step([10,10])