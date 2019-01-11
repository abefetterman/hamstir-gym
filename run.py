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
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

from hamstir_gym.model import NatureLitePolicy, MobilenetPolicy, set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    
    set_global_seeds(args.seed)
    
    env = HamstirRoomEmptyEnv(render=True, dim=192)
    env.seed(args.seed)
    env = DummyVecEnv([lambda: env])
    
    model = PPO2.load(args.model, policy=MobilenetPolicy)
    sess = model.sess
    graph = sess.graph
    input = graph.get_tensor_by_name('model/module_apply_default/hub_input/Sub:0')
    output = graph.get_tensor_by_name('model/pi/add:0')
    
    obs = env.reset()
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            # print(action, sess.run(input, feed_dict={model.act_model.obs_ph:obs}))
            # print(action, sess.run(output, feed_dict={input:obs}))
            obs, rewards, dones, info = env.step(action)
            print(sum(rewards),action, dones)
    except KeyboardInterrupt:
        print('done')