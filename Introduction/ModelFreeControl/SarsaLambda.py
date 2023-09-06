import gymnasium as gym
import numpy as np
from collections import defaultdict

class SarsaLambdaControl:
    
    def __init__(self, env, discount_factor=0.99, epsilon=1, epsilon_decay = 1, final_epsilon = 0.01, alpha = 0.1, alpha_decay = 0.9999, lamb = 0.6):
        
        self.env = env

        # Regarding gamma
        self.discount_factor = discount_factor
        
        # Regarding epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Regarding alpha
        self.alpha = alpha
        self.alpha_decay = alpha_decay

        # Regarding lambda
        self.lamb = lamb

        # Action-Value function and eligibility trace.
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        
    

    def generate_episode(self):
        """
        Generates an episode using current Q-values.
        """
        
        episode = []
        # Env.reset() returns some info (empty dictionary in this case) which needs to be discarded.
        state, _ = env.reset()
        action = self.act(state)
        E = {}

        while True:
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = self.act(next_state)
            episode.append((state, action, reward))
            
            E = defaultdict(int, {key: eligibility * self.lamb * self.discount_factor for key, eligibility in E.items()})
            E[(state, action)] += 1
            
            
            if terminated or truncated:
                
                # If terminated, then we only update q(s, a) based on some sort of TD-error with only immediate reward.
                for sa in E:
                    self.Q[sa[0]][sa[1]] += self.alpha * (reward - self.Q[state][action]) * E[sa]
                
                break
            
            # Updating the q-values in the usual way.
            for sa in E:
                    self.Q[sa[0]][sa[1]] += self.alpha * (reward + self.discount_factor * self.Q[next_state][next_action] - self.Q[state][action]) * E[sa]
            
            # Making things ready for the next step
            state = next_state
            action = next_action

        return episode
    
    def act(self, state):
        """
        Decides which action we should take in a state, according to our policy.
        """

        if (np.random.random() < self.epsilon):
            return np.random.randint(self.env.action_space.n)
        
        else:
            return np.argmax(self.Q[state])


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
        STEP_DECAY = self.epsilon / (num_episodes/2)
        for i in range(1, num_episodes + 1):
            self.generate_episode()
            self.epsilon = max(self.final_epsilon, self.epsilon - STEP_DECAY)
            self.alpha *= self.alpha_decay
            if (i % 10000 == 0):
                print(i)

        return self.get_policy()

    

env = gym.make('Blackjack-v1')
agent = SarsaLambdaControl(env)


policy = agent.learn(1_000_000)
with open('SarsaLambda.txt', 'w') as f:
        sorted_dict = dict(sorted(policy.items(), key = lambda x: (x[0][2], x[0][0], x[0][1])))
        for state in sorted_dict:
            f.write(str(state) + ":" + str(policy[state]) + "\n")
        f.close()
