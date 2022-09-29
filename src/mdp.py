from abc import abstractclassmethod
import numpy as np

from collections import namedtuple
from scipy.spatial.distance import hamming

# def hamming(a, b):
#   return np.sum(np.logical_not(np.logical_xor(a, b)))

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
    def __init__(self, n_antigens, n_antigen_patterns, n_effector_cells, infection_rate) -> None:
        self.N = n_antigens # variety of antigens
        self.P = n_antigen_patterns # number of states
        self.M = n_effector_cells # number of actions
        self.R0 = self.M/2
        self.infection_rate = infection_rate

        self.states = np.random.randint(2, size=(self.N, self.P)) # [number of antigens, number of patterns]
        self.actions = np.random.randint(2, size=(self.M, self.P)) # [number of effector cells, number of patterns]

        self.uninfected_state = self.states[:, 0] # vector of len N
        self.infected_states = self.states[:, 1:]
        
        self.uninfected_action = self.actions[:, 0] # vector of len M
        self.infected_actions = self.actions[:, 1:]

        # intialize state as uninfected -> index is nan
        self.current_state = State(False, np.nan)

    def reward(self, action):
        """ reward function:
        This functional form indicates
        that the immune system receives the highest reward M when
        the activity pattern a matches the most effective one for the
        antigen pattern

        :param action: effector cell activation pattern
        :type action: binary array
        :return: state-action reward
        :rtype: array
        """
        if self.current_state.is_infected:
            effective_action = self.infected_actions[:, self.current_state.idx]
        else:
            effective_action = self.uninfected_action
        # hamming returns a % so need to multiply by length of array
        return self.M - hamming(effective_action, action)*len(action)

    def optimal_reward(self):
        # optimal reward is obtained when hamming function is equal to 0
        return self.M

    def initial_state(self):
        """Initialize state

        :return: infected or uninfected state depending on initialization
        :rtype: np.ndarray
        """
        if self.current_state.is_infected:
            return self.infected_states[:, self.current_state.idx]
        else:
            return self.uninfected_state

    def new_state(self, action):
        """ Transition function from state s(t) to s(t+1)

        :param action: effector cell activation pattern
        :type action: binary array
        """
        if self.current_state.is_infected:
            # already infected, determin transition probability to go back uninfected
            effective_action = self.infected_actions[:, self.current_state.idx]
            # transition_prob_thresh = max([0, (self.M - hamming(effective_action, action)*len(action) - self.R0)/(self.M - self.R0)])
            transition_prob_thresh = max([0, (hamming(effective_action, action)*len(action))/(self.M)])
            # print('thresh:', transition_prob_thresh)
            if np.random.binomial(1, transition_prob_thresh):
                # probability under threshold -> transition to healthy state
                self.current_state = State(False, np.nan)
                return self.uninfected_state
            else:
                # if not, stay at infected state
                return self.infected_states[:, self.current_state.idx]
        else:
            if np.random.binomial(1, self.infection_rate):
                # healthy -> infected (fixed probability)
                self.current_state = State(True, np.random.randint(self.P-1)) # uniform probabilty chosen from P-1 pathogens
                return self.infected_states[:, self.current_state.idx]
            else:
                # healthy -> healthy
                self.current_state = State(False, np.nan)
                return self.uninfected_state



