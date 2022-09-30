import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

# load last trial
logdir = './logs'
trials = sorted(os.listdir(logdir))

def load_trials(logdir: str, trial_name: str) -> dict:
    trial_path = os.path.join(logdir, trial_name)
    with open(trial_path) as f:
        trial_dict = json.load(f)

    for k, v in trial_dict.items():
        if isinstance(v, list):
            trial_dict[k] = np.array(v)
    
    return trial_dict

def plot_trial_results(trial_dict: dict):
    """ Plot a sample of clone sizes evolution with training step

    :param trial_dict: dictionary of logged values during learning
    :type trial_dict: dict
    """

    sample_idx = np.random.choice(trial_dict['clone_size'].shape[0], 50)
    # clone_size_sample = trial_dict['clone_size'][sample_idx, :]

    epochs = np.arange(1, trial_dict['clone_size'].shape[1]+1)
    fig, ax = plt.subplots(figsize=(7, 10), ncols=1, nrows=2)

    # clone size v epochs
    ax = ax.flatten()
    for i in sample_idx:
        ax[0].plot(epochs, trial_dict['clone_size'][i, :])
    
    ax[0].set_xlabel('epochs (x100)')
    ax[0].set_ylabel('normalized clone size')

    # reward v epochs
    ax[1].plot(epochs, trial_dict['reward'])
    ax[1].set_xlabel('epochs (x100)')
    ax[1].set_ylabel('reward')

    plt.show()



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Plot clone size trial')

    parser.add_argument('trial_name', metavar='trial_name', type=str, 
        help='name of the JSON trial file')

    args = parser.parse_args()

    trial_dict = load_trials(logdir, args.trial_name)
    plot_trial_results(trial_dict)