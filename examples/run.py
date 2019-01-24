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
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--debug_video', type=str, help='path to save debug video')
    parser.add_argument('--robot_eye_video', type=str, help='path to save robot-eye video')
    parser.add_argument('--verbose', action='store_true', help='show reward, actions, dones for each step')
    args = parser.parse_args()
    
    if args.robot_eye_video:
        import av
        output = av.open(args.robot_eye_video,mode='w')
        stream = output.add_stream('mpeg4', rate=13)
        stream.pix_fmt = 'yuv420p'
        stream.height, stream.width = 128, 128
    
    set_global_seeds(args.seed)
    
    env = HamstirRoomEmptyEnv(render=True, dim=128)
    if args.debug_video:
        env.logVideo(args.debug_video)
    env.seed(args.seed)
    env = DummyVecEnv([lambda: env])
    
    model = PPO2.load(args.model, policy=NatureLitePolicy)
    sess = model.sess
    graph = sess.graph
    # input = graph.get_tensor_by_name('model/module_apply_default/hub_input/Sub:0')
    # output = graph.get_tensor_by_name('model/pi/add:0')
    
    obs = env.reset()
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            # print(action, sess.run(input, feed_dict={model.act_model.obs_ph:obs}))
            # print(action, sess.run(output, feed_dict={input:obs}))
            obs, rewards, dones, info = env.step(action)
            if args.verbose:
                print(sum(rewards),action, dones)
            if args.robot_eye_video:
                frame = av.VideoFrame.from_ndarray(obs[0], format='rgb24')
                packet = stream.encode(frame)
                output.mux(packet)
    except KeyboardInterrupt:
        print('done')
    finally:
        if args.robot_eye_video:
            output.close()