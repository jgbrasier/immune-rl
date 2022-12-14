import numpy as np
from tqdm import tqdm
import os
import json

from src.agent import Agent
from src.mdp import SingleUninfected

class Environment:

    def __init__(self, logdir, seed=4) -> None:
        self.logdir = logdir
        self.save_dict = dict()
        self.seed = seed
        self._init_logir()

    def _init_logir(self):
        # init logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def _save_trial(self):
        trial_name = 'trial_' + str(len(os.listdir(self.logdir)))
        trial_path = os.path.join(self.logdir, trial_name)

        # convert ndarray to lists
        for k, v in self.save_dict.items():
            if isinstance(v, np.ndarray):
                self.save_dict[k] = v.tolist()

        with open(trial_path+'.json', 'w') as fp:
            json.dump(self.save_dict, fp, sort_keys=True, indent=4)


    def simulate(self, max_epoch, lr, n_states, state_dimension, action_dimension, n_hidden,  record_every=1000) -> None:
        np.random.seed(seed=self.seed)
        # intialise agent and MDP
        agent = Agent(n_hidden, state_dimension, action_dimension)
        mdp = SingleUninfected(n_states, state_dimension, action_dimension, infection_rate=0.5)
        state = mdp.initial_state()
        action = agent.policy(state, 1.0)
        
        self.save_dict['clone_size'] = np.zeros((len(agent.clone_size), max_epoch//record_every))
        self.save_dict['reward'] = np.zeros(max_epoch//record_every)
        self.save_dict['state'] = np.zeros((len(state), max_epoch//record_every))
        self.save_dict['action'] = np.zeros((len(action), max_epoch//record_every))
        self.save_dict['w'] = agent.w
        self.save_dict['u'] = agent.u
 
        #beta = 5.0 
        # # linearly scale beta from 1.0 to 20.0 with epochs
        beta = np.linspace(1.0, 20.0, max_epoch)
        self.save_dict['beta'] = beta

        # run trial
        for epoch in tqdm(range(max_epoch)):
            action = agent.policy(state, beta[epoch])
            reward = mdp.reward(action)
            agent.learn(state, action, reward, lr=lr)
            state = mdp.new_state(action)

            # record every x epochs
            if epoch%record_every == 0:
                idx = epoch//record_every
                # save clone size evolution
                # print('reward:', reward)
                # print('max clone size:', np.max(agent.clone_size))
                # print('min clone size:', np.min(agent.clone_size))
                self.save_dict['clone_size'][:, idx] = agent.clone_size
                self.save_dict['reward'][idx] = reward
                self.save_dict['state'][:, idx] = state

        
        self._save_trial()



