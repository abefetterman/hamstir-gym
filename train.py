import gym
import hamstir_gym
import baselines
from baselines.ppo2 import ppo2
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def env_fn(): 
    return gym.make('hamstir-room-empty-v0')
    
env = DummyVecEnv([env_fn])

ppo2.learn(network='cnn', env=env, total_timesteps=1000000)