# main file to run learning trials


from sim import Environment

max_epoch = 1000
n_clones = 5000
n_antigens = 100
n_antigen_patterns = 30 
n_effector_cells = 20


env = Environment()

env.simulate(max_epoch, n_clones, n_antigens, n_antigen_patterns, n_effector_cells)