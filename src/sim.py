import numpy as np
import tqdm as tqdm

from src.agent import Agent
from src.mdp import SingleUninfected

class Environment:

    def __init__(self) -> None:
        pass

    def simulate(self, max_epoch, n_clones, n_antigens, n_antigen_patterns, n_effector_cells) -> None:

        agent = Agent(n_clones, n_antigen_patterns, n_effector_cells)
        mdp = SingleUninfected(n_antigens, n_antigen_patterns, n_effector_cells, infection_rate=0.9)

        state = agent.initial_state()

        beta = 5.0 # linearly scale from 1.0 to 20.0 with epochs

        for epoch in tqdm(range(max_epoch)):
            action = agent.policy(state, beta)
            reward = mdp.reward(action)
            agent.learn(state, action, reward, lr=0.001)
            state = mdp.new_state(action)

            print(reward)

    
