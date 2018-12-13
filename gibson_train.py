import gym
import hamstir_gym
import baselines
from baselines.ppo2 import ppo2
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from hamstir_gym.envs.hamstir_gibson_env import HamstirGibsonEnv
from gibson.utils.fuse_policy2 import CnnPolicy2
import os
import argparse

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'gibson_train_allensville.yaml')
print(config_file)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()
    

    env = HamstirGibsonEnv(config = args.config)
    env = DummyVecEnv([lambda: env])
    
    ppo2.learn(policy=CnnPolicy2, env=env, total_timesteps=int(1e7), \
                    log_interval = 1, save_interval = 10, gamma=0.95)