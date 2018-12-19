from hamstir_gym.envs.hamstir_gibson_env import HamstirGibsonEnv
from hamstir_gym.envs.hamstir_room_empty_env import HamstirRoomEmptyEnv
import argparse
import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO2

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'gibson_train_allensville.yaml')
print(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()
    
    env = HamstirGibsonEnv(config=args.config)
    env = DummyVecEnv([lambda: env])
    
    model = PPO2.load(args.model)
    
    obs = env.reset()
    try:
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
    except:
        print('done')