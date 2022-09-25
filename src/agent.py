import numpy as np

sigmoid = lambda x: 1/(1+np.exp(-x))

class Agent:
    """ An agent interacts with the environment
    it follows a policy to generate an action in response to a state obeservation.

    """

    def __init__(self, n_clones, n_antigens, n_effector_cells):
        
        self.K = n_clones
        self.N = n_antigens
        self.M = n_effector_cells
        self.clone_size = np.ones((self.K, ))
        # Th clone <-> antigen interaction
        # affinity of a TCR towards antigen should be sparse
        # 1 TCR only reacts with a small number of antigens
        self.interaction_strength = np.random.normal(scale=np.sqrt(2/self.N), size=(self.K, self.N))
        # Stimulus of effector cell to clone
        self.stimulus_strength = np.random.normal(scale=np.sqrt(2/self.K), size=(self.M, self.K))

        self.activity = None

    def _activity(self, state):
        """Binary Th cell activity depending on an input state

        :param state: antigen pattern
        :type state: binary array
        :return: clone activity
        :rtype: array
        """ 
        return sigmoid(np.dot(self.interaction_strength, state)) # [K, N] x [N, 1] -> [K, 1]

    def policy(self, state, beta):
        """ Policy for effector cell activation

        :param state: antigen pattern
        :type state: binary array
        :param beta: cost scaling parameters
        :type beta: float
        :return: effector cell activation map
        :rtype: array
        """
        activity = self._activity(state)
        # stimulus_strength: [M, K]
        # activity: [K, 1]
        proba = sigmoid(beta*np.dot(self.stimulus_strength, activity))
        # return action: [M, 1]
        return np.random.binomial(1, proba)

    def q_function(self, state, action):
        """ Q function : quality of state-action combination

        :param state: antigen pattern
        :type state: binary array
        :param action: effector cell activation pattern
        :type action: binary array [M, 1]
        :return: q function - conditional probability
        :rtype: float
        """
        activity = self._activity(state)
        # np.dot(stimulus_strength, clones_size * activity): [M, K] * [K, 1] -> [M, 1]
        # np.dot(previous, action): [M, 1] * [M, 1] -> [1, 1] float
        return np.dot(np.dot(self.stimulus_strength, self.clone_size*activity), action)

    def learn(self, state, action, reward, lr=0.001):
        activity = self._activity(state)
        q = self.q_function(state, action)
        lambda_ = (reward - q)*activity*np.dot(self.stimulus_strength.T, action)
        self.clone_size += lr * self.clone_size * lambda_
        self.clone_size = np.maximum(self.clone_size, 0) # ensures positive clone sizes