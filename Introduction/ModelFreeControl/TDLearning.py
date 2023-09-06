import gymnasium as gym
import numpy as np
from collections import defaultdict

class TDLearning:
    
    def __init__(self, env, discount_factor=0.95, epsilon=1, epsilon_decay = 1, final_epsilon = 0.5, alpha = 0.01, alpha_decay = 1):
        self.env = env
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.alpha = alpha
        self.alpha_decay = alpha_decay

        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    

    def generate_episode(self):
        """
        Generates an episode using current Q-values.
        """
        
        episode = []
        # Env.reset() returns some info (empty dictionary in this case) which needs to be discarded.
        state, _ = env.reset()
        action = self.act(state)
        
        while True:
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = self.act(next_state)
            episode.append((state, action, reward))

            
            if terminated or truncated:
                # If terminated, then we only update q(s, a) based on some sort of TD-error with only immediate reward.
                self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
                break
            
            # Updating the q-values in the usual way.
            self.Q[state][action] += self.alpha * (reward + self.discount_factor * self.Q[next_state][next_action] - self.Q[state][action])
            
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
agent = TDLearning(env)
policy = agent.learn(10_000)
with open('Policies/TDlearning.txt', 'w') as f:
        sorted_dict = dict(sorted(policy.items(), key = lambda x: (x[0][2], x[0][0], x[0][1])))
        for state in sorted_dict:
            f.write(str(state) + ":" + str(policy[state]) + "\n")
        f.close()
