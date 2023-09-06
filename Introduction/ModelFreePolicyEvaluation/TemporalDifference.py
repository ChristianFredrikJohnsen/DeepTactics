import numpy as np
from collections import defaultdict


class TemporalDifferenceAgent:
    
    # Initializing an agent which is living in an environemnt.
    # Alpha is the learning rate, and gamma is the discount factor.
    def __init__(self, env, num_episodes, alpha, gamma):
        self.env = env
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.V = defaultdict(float)

    def policy(self, state):
        # Random policy for simplicity
        return np.random.choice(self.env.action_space(state))

    def generate_episode(self):
        episode = []
        state = self.env.reset()

        for t in range(100):  # Limit episode length
            
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        return episode

    def update_values(self, episode):
        
        # In this case, we want to update our value-function based on the TD(0)-algorithm.
        # V(s_t) <- V(s_t) +  α( R_(t+1) + γ*V(s_(t+1) - V(s) )
        # You basically update the state-value function based on a weighted temporal difference error.
        # How much weight which should be added to the TD-error, is determined by the learning rate alpha.

        for t in range(len(episode) - 1):  # Going throught each state in the episode, except for the last one.
            
            # Extracting the useful information.
            state, _, reward = episode[t]
            next_state, _ , _ = episode[t+1]

            # Updating the value-function based on the TD(0) algorithm.
            self.V[state] += self.alpha * ( reward + self.gamma * self.V[next_state] - self.V[state])        

    def learn(self):
        
        for i in range(self.num_episodes):
            episode = self.generate_episode()
            self.update_values(episode)

