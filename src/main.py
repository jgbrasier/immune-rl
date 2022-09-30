# main file to run learning trials
import os
import sys

sys.path.insert(0, os.getcwd())

from src.env import Environment

max_epoch = 100000
lr = 0.1
n_hidden = 5000 # K
n_states = 30 # P
state_dimension = 100 # N
n_actions = 20 # M


env = Environment(logdir='./logs', seed=6)

env.simulate(max_epoch, lr, n_states, state_dimension, n_actions, n_hidden)

