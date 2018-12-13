import gym
import hamstir_gym
import baselines
from baselines.ppo2 import ppo2
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def env_fn(): 
    return gym.make('hamstir-gibson-v0')
    
env = DummyVecEnv([env_fn])

ppo2.learn(network='cnn', env=env, total_timesteps=int(1e7), \
                log_interval = 1, save_interval = 10, gamma=0.95) #, \
                #load_path='./models/ppo2-cnn-160px-0500e.pt')