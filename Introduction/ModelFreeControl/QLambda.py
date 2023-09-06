import gymnasium as gym
import numpy as np
from collections import defaultdict

class QLambdaControl:
    
    def __init__(self, env, discount_factor=0.99, epsilon=1, epsilon_decay = 1, final_epsilon = 0.1, alpha = 0.5, alpha_decay = 0.9999, final_alpha = 0.01, lamb = 0.6, diff_threshold = 10**-2):
        
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
        self.final_alpha = final_alpha

        # Regarding lambda
        self.lamb = lamb

        # Difference threshold to check for convergence
        self.diff_threshold = diff_threshold

        # Action-Value function and eligibility trace.
        self.Q = defaultdict(lambda: np.random.rand(env.action_space.n))
        self.Q_previous = self.Q.copy()
    

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
            episode.append((state, action, reward))
            
            # Updating the eligibility trace:
            if (action == np.argmax(self.Q[state])):
                E = defaultdict(int, {key: eligibility * self.lamb * self.discount_factor for key, eligibility in E.items()})
            else:
                E = defaultdict(int)

            E[(state, action)] += 1

            # If terminated or truncated, we do not pick a next action.
            if terminated or truncated:

                # If terminated, then we only update q(s, a) based on some sort of TD-error with only immediate reward.
                for sa in E:
                    self.Q[sa[0]][sa[1]] += self.alpha * (reward - self.Q[state][action]) * E[sa]
                
                break
            
            # If not terminated or truncated, then we pick a next action:
            next_action = self.act(next_state)
            
            # Updating the q-values in the usual way.
            for sa in E:
                    self.Q[sa[0]][sa[1]] += self.alpha * (reward + self.Q[next_state][np.argmax(self.Q[next_state])] - self.Q[state][action]) * E[sa]
            
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
        STEP_EPSILON_DECAY = self.epsilon / (num_episodes/2)
        STEP_ALPHA_DECAY = self.alpha / (num_episodes  / 2)
        
        for i in range(1, num_episodes + 1):
            self.generate_episode()
            self.epsilon = max(self.final_epsilon, self.epsilon - STEP_EPSILON_DECAY)
            self.alpha = max(self.final_alpha, self.alpha - STEP_ALPHA_DECAY)
            
            # Checking if the policy has converged.
            max_diff = 0
            if (i != 1):
                max_diff = max( abs(self.Q[s][a] - self.Q_previous[s][a]) for s in self.Q if s in self.Q_previous for a in range(self.env.action_space.n) )

            # I assume that it is impossible to get a max_diff of 0 if you are actually learning, so if it is 0, then something is wrong,
            # and we just continue to learn. There might be some mistake in my algorithm, since max_diff of 0 should not happen in theory.
            # Perhaps an episode just ends if dealer gets 21? I don't know. 

            if (max_diff < self.diff_threshold and max_diff != 0):
                break
            
            self.Q_previous = self.Q.copy()
            if (i % 10000 == 0):
                print(i)

        return self.get_policy()

env = gym.make('Blackjack-v1')
agent = QLambdaControl(env)
policy = agent.learn(10_000)

with open('Policies/QLambda.txt', 'w') as f:
        sorted_dict = dict(sorted(policy.items(), key = lambda x: (x[0][2], x[0][0], x[0][1])))
        for state in sorted_dict:
            f.write(str(state) + ":" + str(policy[state]) + "\n")
        f.close()
