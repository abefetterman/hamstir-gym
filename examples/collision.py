import tensorflow as tf
import numpy as np
import gym

from hamstir_gym.envs.hamstir_room_empty_env import HamstirRoomEmptyEnv
from hamstir_gym.model import nature_cnn_lite, set_seed

def model(obs_ph, hidden_dim=32, n_layers=1, n_out=3, return_logits=True):
    with tf.variable_scope('model'):
        net = nature_cnn_lite(obs_ph)
        logits = tf.layers.dense(net, units=n_out, activation=None)
        # `logits` will predict collision--use -logits to sample from 
        actions = tf.squeeze(tf.multinomial(logits=-logits,num_samples=1), axis=1)
    if return_logits:
        return actions, logits
    return actions

def get_loss(logits, act_ph, collision_ph, n_acts=3):
    with tf.variable_scope('loss'):
        action_one_hots = tf.one_hot(act_ph, n_acts)
        logit_action = tf.reduce_sum(action_one_hots * logits, axis=1)
        # logit_correct will be high if prediction is correct
        logit_correct = logit_action * (collision_ph * 2 - 1) 
        loss = -tf.reduce_mean(tf.log_softmax(logit_correct))
    return loss

def train(sess=None,lr=1e-2, gamma=0.99, n_iters=50, batch_size=5000):
    env = HamstirRoomEmptyEnv(render=False, dim=128, step_ratio=50, discrete=True)
    obs_dim = env.observation_space.shape
    n_acts = env.action_space.n

    obs_ph = tf.placeholder(shape=(None, *obs_dim), dtype=tf.float32)
    actions, logits = model(obs_ph, hidden_dim, n_layers, n_acts)

    collision_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    loss = get_loss(logits, act_ph, collision_ph, n_acts)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    if (sess==None):
        sess=tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())

    for i in range(n_iters):
        batch_obs, batch_acts, batch_rtgs, batch_rets, batch_lens = [], [], [], [], []

        obs, rew, done, ep_rews = env.reset(), 0, False, []
        while True:
            batch_obs.append(obs.copy())
            act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]
            obs, rew, done, _ = env.step(act)
            batch_acts.append(act)
            ep_rews.append(rew)
            if done:
                batch_rets.append(sum(ep_rews))
                batch_lens.append(len(ep_rews))
                batch_rtgs.extend(discount_cumsum(ep_rews, gamma))
                obs, rew, done, ep_rews = env.reset(), 0, False, []
                if len(batch_obs) > batch_size:
                    break

        batch_loss, _ = sess.run([loss, train_op], feed_dict={obs_ph: np.array(batch_obs),
                                                              act_ph: np.array(batch_acts),
                                                              adv_ph: np.array(batch_rtgs)})
        print('itr: %d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    try:
        with tf.Session() as sess:
            train(sess)
    except KeyboardInterrupt:
        print('bye')