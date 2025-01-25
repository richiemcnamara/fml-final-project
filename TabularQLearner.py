import numpy as np

class TabularQLearner:

    def __init__ (self, states = 100, actions = 4, alpha = 0.2, gamma = 0.9, epsilon = 0.98, epsilon_decay = 0.999, dyna=0):
        # Store all the parameters as attributes (instance variables).
        # Initialize any data structures you need.
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        self.q_values = np.random.normal(0, 0.01, (self.states, self.actions))
        self.prev_state = None
        self.prev_action = None

        self.dyna = dyna
        self.num_exp = 0
        self.exp_history = {}
        

    def train (self, s, r):
        # Receive new state s and new reward r.  Update Q-table and return selected action.
        # Consider: The Q-update requires a complete <s, a, s', r> tuple.
        #           How will you know the previous state and action?
        
        # Epsilon Greedy
        if np.random.random() <= self.epsilon:
            a = np.random.randint(self.actions)
        else:
            a = self.q_values[s, :].argmax()

        self.epsilon *= self.epsilon_decay

        # Update Q
        self.q_values[self.prev_state, self.prev_action] = ((1 - self.alpha) * self.q_values[self.prev_state, self.prev_action]) + (self.alpha * (r + (self.gamma * self.q_values[s, :].max())))

        # Dyna-Q
        if self.dyna:
            self.exp_history[self.num_exp] = (self.prev_state, self.prev_action, s, r)

            for i in range(self.dyna):
                s_samp, a_samp, s_prime_samp, r_samp = self.exp_history[np.random.randint(self.num_exp + 1)]
                self.q_values[s_samp, a_samp] = ((1 - self.alpha) * self.q_values[s_samp, a_samp]) + (self.alpha * (r_samp + (self.gamma * self.q_values[s_prime_samp, :].max())))

        self.num_exp += 1
        self.prev_state = s
        self.prev_action = a

        return a

    def test (self, s):
        # Receive new state s.  Do NOT update Q-table, but still return selected action.
        #
        # This method is called for TWO reasons: (1) to use the policy after learning is finished, and
        # (2) when there is no previous state or action (and hence no Q-update to perform).
        #
        # When testing, you probably do not want to take random actions... (What good would it do?)

        a = self.q_values[s, :].argmax()

        self.prev_state = s
        self.prev_action = a
        
        return a