import numpy as np

sigmoid = lambda x: 1/(1+np.exp(-x))

class Agent:
    """ An agent interacts with the environment
    it follows a policy to generate an action in response to a state obeservation.

    """

    def __init__(self, n_hidden, state_dimension, action_dimension):
        
        self.K = n_hidden
        self.N = state_dimension
        self.M = action_dimension
        self.clone_size = np.ones((self.K, ))
        # Th clone <-> antigen interaction
        # affinity of a TCR towards antigen should be sparse
        # 1 TCR only reacts with a small number of antigens
        self.w = np.random.normal(scale=np.sqrt(2/self.N), size=(self.K, self.N))
        # Stimulus of effector cell to clone
        self.u = np.random.normal(scale=np.sqrt(2/self.K), size=(self.M, self.K))

        self.activity = None

    def policy(self, state, beta):
        """ Policy for effector cell activation

        :param state: antigen pattern
        :type state: binary array
        :param beta: cost scaling parameters
        :type beta: float
        :return: effector cell activation map
        :rtype: array
        """
        self.h = sigmoid(np.dot(self.w, state)) # [K, 1]
        self.nh = self.clone_size* self.h # [K, 1]
        action = np.random.binomial(1, sigmoid(beta * np.dot(self.u, self.nh))) # [M, K] * [K, 1] -> [M, 1]
        return action

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

    def learn(self, state, action, reward, lr=0.1):
        q = np.dot(np.dot(self.u, self.nh), action)
        lam = (reward - q) * self.h * np.dot(self.u.T, action)
        self.clone_size+= lr * self.clone_size* lam
        self.clone_size= np.maximum(self.clone_size, 0.0) # To ensure n >= 0

