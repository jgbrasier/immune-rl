import numpy as np

sigmoid = lambda x: 1/(1+np.exp(-x))

class Agent:
    # agent interacts with environement
    # follows a policy to generate an action in response to observing a state

    def __init__(self, n_clones, n_antigen_patterns, n_effector_cells):
        
        self.P = n_antigen_patterns
        self.M = n_effector_cells
        self.clone_size((n_clones, ))
        # Th clone <-> antigen interaction
        # affinity of a TCR towards antigen should be sparse
        # 1 TCR only reacts with a small number of antigens
        self.interaction_strength = np.random.normal(scale=np.sqrt(2/self.P), size=(n_clones, self.P))
        # Stimulus of effector cell to clone
        self.stimulus_strength = np.random.normal(scale=np.sqrt(2/n_clones), size=(self.M, n_clones))

        self.activity = None

    def _activity(self, state):
        return sigmoid(np.dot(self.interaction_strength, state))

    def policy(self, state, beta):
        activity = self._activity(state)
        proba = sigmoid(beta*np.dot(self.clone_size, activity))
        # return action
        return np.random.binomial(1, proba)

    def q_function(self, state, action):
        activity = self._activity(state)
        return np.dot(np.dot(self.clone_size, activity), np.dot(self.stimulus_strength, action))

    def learn(self, state, action, reward, lr=0.001):
        activity = self._activity(state)
        q = self.q_function(state, action)
        lambda_ = (reward - q)*activity*np.dot(self.stimulus_strength.T, action)
        self.clone_size += lr * self.clone_size * lambda_
        self.clone_size = np.maximum(self.clone_size, 0) # ensures positive clone sizes