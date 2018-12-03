import copy
import gym
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from scipy.misc import imresize





##### testing only
# MAX_EXPERIENCES = 10000
# MIN_EXPERIENCES = 1000


MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 80
K = 4 #env.action_space.n




def downsample_image(A):
  #B = A[31:195] # select the important parts of the image
  B = B.mean(axis=2) # convert to grayscale

  # downsample image
  # changing aspect ratio doesn't significantly distort the image
  # nearest neighbor interpolation produces a much sharper image
  # than default bilinear
  B = imresize(B, size=(IM_SIZE, IM_SIZE), interp='nearest')
  return B


def update_state(state, obs):
  obs_small = downsample_image(obs)
  return np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)


class DQN:
  def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, gamma, scope):

    self.K = K
    self.scope = scope

    with tf.variable_scope(scope):

      # inputs and targets
      self.X = tf.placeholder(tf.float32, shape=(None, 4, IM_SIZE, IM_SIZE), name='X')

      # tensorflow convolution needs the order to be:
      # (num_samples, height, width, "color")
      # so we need to tranpose later
      self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
      self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

      # calculate output and cost
      # convolutional layers
      # these built-in layers are faster and don't require us to
      # calculate the size of the output of the final conv layer!
      Z = self.X / 255.0
      Z = tf.transpose(Z, [0, 2, 3, 1])
      for num_output_filters, filtersz, poolsz in conv_layer_sizes:
        Z = tf.contrib.layers.conv2d(
          Z,
          num_output_filters,
          filtersz,
          poolsz,
          activation_fn=tf.nn.relu
        )

      # fully connected layers
      Z = tf.contrib.layers.flatten(Z)
      for M in hidden_layer_sizes:
        Z = tf.contrib.layers.fully_connected(Z, M)

      # final output layer
      self.predict_op = tf.contrib.layers.fully_connected(Z, K)

      selected_action_values = tf.reduce_sum(
        self.predict_op * tf.one_hot(self.actions, K),
        reduction_indices=[1]
      )

      cost = tf.reduce_mean(tf.square(self.G - selected_action_values))
      # self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
      # self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
      # self.train_op = tf.train.RMSPropOptimizer(2.5e-4, decay=0.99, epsilon=10e-3).minimize(cost)
      self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)
      # self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
      # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

      self.cost = cost

  def copy_from(self, other):
    mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
    mine = sorted(mine, key=lambda v: v.name)
    theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
    theirs = sorted(theirs, key=lambda v: v.name)

    ops = []
    for p, q in zip(mine, theirs):
      actual = self.session.run(q)
      op = p.assign(actual)
      ops.append(op)

    self.session.run(ops)

  def set_session(self, session):
    self.session = session

  def predict(self, states):
    return self.session.run(self.predict_op, feed_dict={self.X: states})

  def update(self, states, actions, targets):
    c, _ = self.session.run(
      [self.cost, self.train_op],
      feed_dict={
        self.X: states,
        self.G: targets,
        self.actions: actions
      }
    )
    return c

  def sample_action(self, x, eps):
    if np.random.random() < eps:
      return np.random.choice(self.K)
    else:
      x = np.reshape(x, (4,80,80))
      return np.argmax(self.predict([x])[0])