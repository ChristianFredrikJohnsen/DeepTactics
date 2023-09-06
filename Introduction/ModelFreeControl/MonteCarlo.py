import gymnasium as gym
import numpy as np
from collections import defaultdict

class MonteCarloControl:
    
    def __init__(self, env, discount_factor=1.0, epsilon=0.5, epsilon_decay = 0.99999):
        self.env = env
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(env.action_space.n))

    

    def generate_episode(self):
        """
        Generates an episode using current Q-values.
        """
        
        episode = []
        # Env.reset returns some info (empty dictionary in this case) which needs to be discarded.
        state, _ = env.reset()
        
        while True:
            
            action = self.act(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))

            if terminated or truncated:
                break
            
            state = next_obs

        return episode
    
    def act(self, state):
        """
        Decides which action we should take in a state, according to our policy.
        """

        if (np.random.random() < self.epsilon):
            return np.random.randint(self.env.action_space.n)
        
        else:
            return np.argmax(self.Q[state])

    def update_q_values(self, episode):
        """
        Updates the action-value function estimate using the episode history.
        """
        
        # I am assuming that the last entry in the episode only has one bit of info, which is the terminal state.
        # It doesn't make sense that the terminal state has an action and a reward.
        # I might be wrong. 

        G = 0
        visited_state_actions = set()

        for t in reversed(range(len(episode))):  # Loop through the episode
            
            state, action, reward = episode[t]
            G = reward + self.discount_factor * G
            #sa = tuple([state, action])

            if (state, action) not in visited_state_actions:
                
                visited_state_actions.add((state, action))
                self.N[state][action] += 1

                # Updating the action-value-function estimate, with the running average method.
                # µ_k = µ_(k-1) + 1/k * (x_k - µ_(k-1))
                self.Q[state][action] += (1 / self.N[state][action]) * (G - self.Q[state][action] )


    def get_policy(self):
        """
        Returns the current policy based on the action-value function.
        """
        policy = defaultdict(int)
        for state in self.Q:
            policy[state] = np.argmax(self.Q[state])

        return policy


    def learn(self, num_episodes):
        
        """
        Executes a number of episodes of on-policy learning through sampling.
        """
        
        for i in range(1, num_episodes + 1):
            episode = self.generate_episode()
            self.update_q_values(episode)
            self.epsilon *= self.epsilon_decay
            if (i % 10000 == 0):
                print(i)


        return self.get_policy()

    


env = gym.make('Blackjack-v1')
agent = MonteCarloControl(env)
policy = agent.learn(10000)
with open('Policies/MonteCarloLearning.txt', 'w') as f:
        sorted_dict = dict(sorted(policy.items(), key = lambda x: (x[0][2], x[0][0], x[0][1])))
        for state in sorted_dict:
            f.write(str(state) + ":" + str(policy[state]) + "\n")
        f.close()
