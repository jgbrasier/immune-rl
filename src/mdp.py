from abc import abstractclassmethod
import numpy as np

from collections import namedtuple
from scipy.spatial.distance import hamming


class MDP:
    @abstractclassmethod
    def initial_state(self):
        raise NotImplementedError
    @abstractclassmethod
    def new_state(self, action):
        raise NotImplementedError()
    @abstractclassmethod
    def reward(self, action):
        raise NotImplementedError()
    @abstractclassmethod
    def optimal_reward(self):
        raise NotImplementedError()

State = namedtuple('State', ['is_infected', 'idx'])

class SingleUninfected(MDP):
    """MDP model with P-1 infected states that trasition between a single uninfected state
    """
    def __init__(self, n_antigens, n_antigen_patterns, n_effector_cells) -> None:
        self.N = n_antigens # variety of antigens
        self.P = n_antigen_patterns # number of states
        self.M = n_effector_cells # number of actions

        self.states = np.random.randint(2, size=(self.N, self.P))
        self.actions = np.random.randint(2, size=(self.M, self.P))

        self.uninfected_state = self.states[0, :]
        self.infected_states = self.states[1:, :]
        
        self.uninfected_action = self.actions[0, :]
        self.infected_actions = self.actions[1:, :]

        self.current_state = State(False, np.nan)

    def initial_state(self):
        """Initialize state

        :return: infected or uninfected state depending on initialization
        :rtype: np.ndarray
        """
        if self.current_state.is_infected:
            return self.infected_states[:, self.current_state.idx]
        else:
            return self.uninfected_state

