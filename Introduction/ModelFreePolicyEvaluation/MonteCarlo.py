# Monte Carlo learning is about calculating the average future expected reward from a given state.
# No bias but high variance.
# Updating the current value-function for state s by applying the formula:
# µ_k = µ_(k-1) + 1/k * (x_k - µ_(k-1))
# In this formula, µ is the average expected future reward with k samples.
# x_(k+1) is sample number (k+1). It is the (k+1)'th value of expected future reward you are entering.
# I will try to implement first-visit Monte Carlo.

# I will need some value-function, and i need some sample episodes. 


import numpy as np
from collections import defaultdict


class MonteCarloAgent:
    
    def __init__(self, env, num_episodes, gamma):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.V = defaultdict(float)
        self.N = defaultdict(int)

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
        
        G = 0
        
        visited_states = set()
        
        for t in range(len(episode) - 1, -1, -1):  # Loop backwards through episode
            
            state, _, reward = episode[t]
            G = reward + self.gamma * G

            if state not in visited_states:
                
                visited_states.add(state)
                self.N[state] += 1

                # Updating the value-function estimate, with the running average method.
                # µ_k = µ_(k-1) + 1/k * (x_k - µ_(k-1))
                self.V[state] += (1 / self.N[state]) * (G - self.V[state] )
               

    def learn(self):
        
        for i in range(self.num_episodes):
            episode = self.generate_episode()
            self.update_values(episode)


