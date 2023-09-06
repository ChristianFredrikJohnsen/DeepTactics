import numpy as np
from collections import defaultdict


class TemporalDifferenceLambdaAgent:
    
    '''
    This class is supposed to update the values of the value-function based on the TD(λ) algorithm.
    I will be implementing backward-view TD(λ), as that seems to be the superior algorithm.
    
    For backward-view TD(λ), we have the following formula:
    V(s) <- V(s) + alpha * delta_t * E_t(s)
    In this formula, V(s) is the state-value approximation for state s, alpha is the learning rate,
    delta_t is the temporal difference for this timestep, and E_t(s) is the eligibility trace at timestep t.

    delta_t = ( R_t + gamma * v( s_(t+1) ) - v(s) )
    
    The hard part about this algorithm is getting the eligibility trace right.
    E_t(s) = gamma * lambda * E_(t - 1)(s) + 1(S_t = s)
    '''

    def __init__(self, env, num_episodes, alpha, gamma, lamb):
        
        '''
        Initializing an agent which is living in an environemnt.
        Alpha is the learning rate, gamma is the discount factor, and lamb is the value for lambda in this TD(λ) algorithm.
        λ = 0 leads to TD(0), while λ = 1 leads to Monte Carlo evaluation. You want to empirically test different values of lambda between 0 and 1,
        and just look at what works for your problem.
        '''

        self.env = env
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.V = defaultdict(float)
        self.E = None

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

        self.E = defaultdict(float)

        # It shall be said that GPT4 disagrees with this implementation, but i disagree with GPT4, and we don't get anywhere.
        # I don't have the blessing of the chatbot, but i believe this to be correct.

        for t in range(len(episode) - 1):  # Going through each state in the episode, except for the last one.
            
            # Extracting the useful information.
            state, _, reward = episode[t]
            next_state, _ , _ = episode[t+1]

            # Finding the TD-error.
            TD_error =  reward + self.gamma * self.V[next_state] - self.V[state]

            for s in self.V.keys():
                
                if (self.E[s] == 0):
                    continue

                if (s == state):
                    self.E[s] += 1

                else:
                    self.E[s] *= self.gamma * self.lamb

                self.V[s] += self.alpha * TD_error * self.E[s]

    def learn(self):
        
        for i in range(self.num_episodes):
            episode = self.generate_episode()
            self.update_values(episode)

