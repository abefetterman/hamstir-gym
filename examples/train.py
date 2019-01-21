from hamstir_gym.envs.hamstir_gibson_env import HamstirGibsonEnv
from hamstir_gym.envs.hamstir_room_empty_env import HamstirRoomEmptyEnv
import argparse
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from hamstir_gym.model import NatureLitePolicy, MobilenetPolicy, set_seed
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

best_mean_reward, n_steps = -np.inf, 0

def make_env(log_dir, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param log_dir: (str) location for logging
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    env_log_dir = log_dir # os.path.join(log_dir, str(rank))
    os.makedirs(env_log_dir, exist_ok=True)
    def _init():
        env = HamstirRoomEmptyEnv(render=False, dim=192)
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
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--ncpu', type=int, default=1, help='number of cpus')
    parser.add_argument('--log_dir', type=str, default='/tmp/gym/', help='path for logs')
    parser.add_argument('--tensorboard_dir', type=str, default='../tensorboard', help='path for tensorboard logs')
    parser.add_argument('--load_model', type=str, help='load a model to continue training')
    args = parser.parse_args()
    
    # Create log dir
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    env = DummyVecEnv([make_env(args.log_dir, i, args.seed) for i in range(args.ncpu)])
    
    set_seed(args.seed)
    
    model = PPO2(NatureLitePolicy, env, verbose=1, gamma=0.99, n_steps=2000, tensorboard_log=args.tensorboard_dir)
    if args.load_model:
        model = PPO2.load(args.load_model, policy=NatureLitePolicy)
    
    print('graph seed:', model.graph.seed)
    
    model.learn(total_timesteps=int(1e7), callback=callback, seed=args.seed)
