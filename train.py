from hamstir_gym.envs.hamstir_gibson_env import HamstirGibsonEnv
from hamstir_gym.envs.hamstir_room_empty_env import HamstirRoomEmptyEnv
import argparse
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

best_mean_reward, n_steps = -np.inf, 0

def seedPolicy(seed=None):
class CustomPolicy(CnnPolicy):
    def __init__(self, *args, **kwargs):
        if seed != None:
            set_global_seeds(seed)
        super(CustomPolicy, self).__init__(*args, **kwargs)
return CustomPolicy

def make_env(log_dir, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param log_dir: (str) location for logging
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    env_log_dir = os.path.join(log_dir, str(rank))
    os.makedirs(env_log_dir, exist_ok=True)
    def _init():
        env = HamstirRoomEmptyEnv(render=False)
        if seed != None:
            env.seed(seed + rank)
        env = Monitor(env, env_log_dir, allow_early_resets=True)
        return env
    return _init
    
def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  log_dir = "/tmp/gym/"
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
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ncpu', type=int, default=1)
    args = parser.parse_args()
    
    # Create log dir
    log_dir = "/tmp/gym/"
    tensorboard_dir = "~/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    env = DummyVecEnv([make_env(log_dir, i, args.seed) for i in range(args.ncpu)])
    
    model = PPO2(seedPolicy(args.seed), env, verbose=1, gamma=0.95, n_steps=2000, tensorboard_log=tensorboard_dir)
    
    # set_global_seeds(args.seed)
    print(model.graph.seed)
    # print(env.config)
    
    model.learn(total_timesteps=int(1e7), callback=callback, seed=args.seed)
