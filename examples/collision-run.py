import tensorflow as tf
import numpy as np
import gym

from hamstir_gym.envs.hamstir_room_empty_env import HamstirRoomEmptyEnv
from hamstir_gym.model import nature_cnn_lite, set_seed

def model(obs_ph, n_out=3, return_logits=True):
    with tf.variable_scope('model'):
        net = nature_cnn_lite(obs_ph)
        logits = tf.layers.dense(net, units=n_out, activation=None)
        # `logits` will predict NO collision--use logits to sample from 
        actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
    if return_logits:
        return actions, logits
    return actions

def get_loss(logits, act_ph, collision_ph, n_acts=3):
    with tf.variable_scope('loss'):
        action_one_hots = tf.one_hot(act_ph, n_acts)
        logit_action = tf.reduce_sum(action_one_hots * logits, axis=1)
        # logit_correct will be high if prediction is correct
        # `logits` high predicts NO collision
        logit_correct = logit_action * (1 - collision_ph * 2) 
        loss = -tf.reduce_mean(tf.nn.log_softmax(logit_correct))
        tf.summary.scalar('loss', loss)
    return loss

def train(sess=None,lr=1e-3, gamma=0.99, n_iters=500, horizon=500, rollouts=200):
    env = HamstirRoomEmptyEnv(render=True, dim=128, step_ratio=50, full_reset=False,
                                discrete=True, maxSteps=horizon+2, vel_range=(0.8,1))
    obs_dim = env.observation_space.shape
    n_acts = env.action_space.n

    obs_ph = tf.placeholder(shape=(None, *obs_dim), dtype=tf.float32)
    actions, logits = model(obs_ph, n_acts)

    collision_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    loss = get_loss(logits, act_ph, collision_ph, n_acts)
    train_op = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9,use_nesterov=True).minimize(loss)

    saver = tf.train.Saver()
    running_loss = 0
    running_loss_horizon = 10
    running_loss_best = float('inf')
    
    if (sess==None):
        sess=tf.InteractiveSession()

    saver.restore(sess, "./models/model.ckpt")

    for i in range(n_iters):
        batch_obs, batch_acts, batch_collisions, batch_lens = [], [], [], []

        for _ in range(rollouts):
            obs, rew, done, ep_rews = env.reset(), 0, False, []
            act = np.random.randint(0,n_acts)
            obs, rew, done, _ = env.step(act)
            if done:
                # somehow ended after first action, don't count this one
                break
            batch_obs.append(obs.copy())
            batch_acts.append(act)
            has_collision = 0.0
            max_step = horizon
            for step in range(horizon):
                act,log = sess.run([actions, logits], {obs_ph: np.expand_dims(obs, 0)})
                print(log)
                obs, rew, done, _ = env.step(act[0])
                if done:
                    has_collision = 1.0
                    max_step = step
                    break
            # print('action {} collision {}'.format(batch_acts[-1], has_collision))
            batch_collisions.append(has_collision)
            batch_lens.append(max_step)
            
        batch_loss, _ = sess.run([loss, train_op], feed_dict={obs_ph: np.array(batch_obs),
                                                              act_ph: np.array(batch_acts),
                                                              collision_ph: np.array(batch_collisions)})
        
        running_loss_count = min(i,running_loss_horizon)
        running_loss = (running_loss * running_loss_count + batch_loss) / (running_loss_count + 1)

        print('itr: %d \t loss: %.3f (%.3f) \t ep_len: %.3f'%
                (i, batch_loss, running_loss, np.mean(batch_lens)))
        
if __name__ == '__main__':
    try:
        with tf.Session() as sess:
            train(sess)
    except KeyboardInterrupt:
        print('bye')