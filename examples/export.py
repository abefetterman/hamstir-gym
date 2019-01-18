import argparse
import os
import tensorflow as tf
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2

from hamstir_gym.model import NatureLitePolicy, MobilenetPolicy, set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--graph_out', type=str)
    args = parser.parse_args()
    
    model = PPO2.load(args.model, policy=NatureLitePolicy)
    sess = model.sess
    graph = sess.graph
    
    output_graph_def = tf.graph_util.convert_variables_to_constants( \
      sess, graph.as_graph_def(), ['model/pi/add'])
    
    with tf.gfile.FastGFile(args.graph_out, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    # 
    # train_saver = tf.train.Saver(model.params)
    # train_saver.save(sess, './check.out')