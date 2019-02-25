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
        loss = -tf.reduce_mean(tf.math.log_sigmoid(logit_correct))
        tf.summary.scalar('loss', loss)
    return loss

def train(sess=None,lr=1e-4, gamma=0.99, n_iters=2000, horizon=4, maxEpLen=40, stepsPerEpoch=200):
    env = HamstirRoomEmptyEnv(render=False, dim=128, step_ratio=50, full_reset=False,
                                discrete=True, maxSteps=maxEpLen+2, vel_range=(0.8,1))
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

    sess.run(tf.global_variables_initializer())

    for i in range(n_iters):
        batch_obs, batch_acts, batch_collisions, batch_lens = [], [], [], []

        while len(batch_obs) < stepsPerEpoch:
            obs, rew, done, ep_rews = env.reset(), 0, False, []
            has_collision = 0.0
            max_step = maxEpLen
            for step in range(maxEpLen):
                act = sess.run(actions, {obs_ph: np.expand_dims(obs, 0)})[0]
                if step % horizon == 0:
                    act = np.random.randint(0,n_acts)
                    batch_obs.append(obs)
                    batch_acts.append(act)
                obs, rew, done, _ = env.step(act)
                if done:
                    batch_collisions.append(1.0)
                    max_step = step
                    break
                elif step % horizon == horizon - 1:
                    batch_collisions.append(0.0)
            # print('action {} collision {}'.format(batch_acts[-1], has_collision))
            # batch_collisions.append(has_collision)
            batch_lens.append(max_step)
            
        batch_loss, _ = sess.run([loss, train_op], feed_dict={obs_ph: np.array(batch_obs),
                                                              act_ph: np.array(batch_acts),
                                                              collision_ph: np.array(batch_collisions)})
        
        running_loss_count = min(i,running_loss_horizon)
        running_loss = (running_loss * running_loss_count + batch_loss) / (running_loss_count + 1)

        print('itr: %d \t loss: %.3f (%.3f) \t ep_len: %.3f'%
                (i, batch_loss, running_loss, np.mean(batch_lens)))
                
        if running_loss < running_loss_best:
            running_loss_best = running_loss
            save_path = saver.save(sess, "/tmp/gym/model.ckpt")
            print('New best model saved to %s' % save_path)
        
if __name__ == '__main__':
    try:
        with tf.Session() as sess:
            train(sess)
    except KeyboardInterrupt:
        print('bye')