# main file to run learning trials


from src.sim import Environment

max_epoch = 1000
n_clones = 5000 # K
n_antigens = 100 # N
n_antigen_patterns = 30 # P
n_effector_cells = 20 # M


env = Environment()

env.simulate(max_epoch, n_clones, n_antigens, n_antigen_patterns, n_effector_cells)