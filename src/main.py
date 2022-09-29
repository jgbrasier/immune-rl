# main file to run learning trials
import os
import sys

sys.path.insert(0, os.getcwd())

from src.env import Environment

max_epoch = 100000
lr = 0.1
n_clones = 5000 # K
n_antigens = 100 # N
n_antigen_patterns = 30 # P
n_effector_cells = 20 # M


env = Environment(logdir='./logs')

env.simulate(max_epoch, lr, n_clones, n_antigens, n_antigen_patterns, n_effector_cells)

