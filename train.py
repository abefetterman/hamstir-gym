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

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Evaluate policy performance
  x, y = ts2xy(load_results(log_dir), 'timesteps')
  if len(x) > 0:
      mean_reward = np.mean(y[-100:])
      print(x[-1], 'timesteps')
      print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

      # New best model, you could save the agent here
      if mean_reward > best_mean_reward:
          best_mean_reward = mean_reward
          # Example for saving best model
          print("Saving new best model")
          _locals['self'].save(log_dir + 'best_model.pkl')
  return True



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Create log dir
    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    env = HamstirRoomEmptyEnv()
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    
    model = PPO2(CnnPolicy, env, verbose=1, gamma=0.95, n_steps=2000)
    
    # print(env.config)
    model.learn(total_timesteps=1000000, callback=callback)
