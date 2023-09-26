from collections import defaultdict
import numpy as np
import gymnasium as gym
import pickle
from pathlib import Path

class Easy_Agent():

    def __init__(self, ac_dim, ob_dim, lr, epsilon, gamma, env, episodes, eval_frequency):
        
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = env
        self.episodes = episodes
        self.eval_frequency = eval_frequency

        self.q_values = defaultdict(lambda: [0] * self.ac_dim)
        self.num_updates = 0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return self.exploration_action(state)
        else:
            return self.greedy_action(state)

    def exploration_action(self, state):
        return np.random.randint(0, self.ac_dim)

    def greedy_action(self, state):
        return np.argmax(self.q_values[state])
    
    def update_q_values(self, state, reward, action, next_state, done):
        self.q_values[state][action] += self.lr * (reward + self.gamma * np.max(self.q_values[next_state]) * (1-done) - self.q_values[state][action])
        self.num_updates += 1

    def decay_lr(self, episode):
        self.lr = self.lr if episode % 2_000 != 0 else self.lr * 0.5

    def decay_epsilon(self, episode):
        self.epsilon = self.epsilon if episode % 2_000 != 0 else self.epsilon * 0.8
    
    def save(self, path = "Basic_Reinforcment_Learning/cart_values.pkl"):
        qvals = dict(self.q_values)
        with open (path, 'wb') as f:
            pickle.dump(qvals, f)

    def load(self, path = "Basic_Reinforcment_Learning/cart_values.pkl"):
        with open(path, 'rb') as f:
            values = pickle.load(f)

        # Set up the q-values for the new agent.
        self.q_values = defaultdict(lambda: [0] * self.ac_dim)
        self.q_values.update(values)

    def discretize(self, state):
        return tuple(np.round(state, 1))

    def train_agent(self):
        
        train_env = gym.make(self.env)
        eval_env = gym.make(self.env, render_mode = "human")

        for episode in range(1, self.episodes + 1):

            state, _ = train_env.reset()
            total_reward = 0

            while True:

                action = self.act(self.discretize(state))
                next_state, reward, terminated, truncated, _ = train_env.step(action)

                done = terminated or truncated
                self.update_q_values(self.discretize(state), reward, action, self.discretize(next_state), done)

                state = next_state
                total_reward += reward

                if terminated or truncated:
                    break
            
            self.decay_epsilon(episode)
            self.decay_lr(episode)
            

            if episode % self.eval_frequency == 0:

                obs, _ = eval_env.reset()
                episode_return = 0
                episode_length = 0
                


                while True:
                    action = self.greedy_action(self.discretize(obs))
                    next_obs, reward, terminated, truncated, _ = eval_env.step(action)

                    obs = next_obs
                    episode_return += reward
                    episode_length += 1

                    ## terminated betyr at spillet over, truncated betyr at spillets lengde har blitt nådd. 
                    if terminated or truncated:
                        break 
                print(episode_return)

                print(f'Number of updates: {self.num_updates}')
                print(f'Episode: {episode}, Total reward: {total_reward}')

    def show_skills(self, rounds):
        
        env = gym.make(self.env, render_mode = "human")
        for i in range(rounds):
            
            obs, _ = env.reset()
            total_reward = 0

            while True:
                
                action = self.greedy_action(self.discretize(obs))
                next_obs, reward, terminated, truncated, _ = env.step(action)
                obs = next_obs
                total_reward += reward
                if terminated or truncated:
                    break
        
            print(f'Total reward: {total_reward}')

james = Easy_Agent(ac_dim = 2, ob_dim = 4, lr = 0.8, epsilon = 0.4, gamma = 0.99, env = "CartPole-v1", episodes = 1, eval_frequency = 1_000)
# james.train_agent()
# james.save()
# print(james.q_values[(0, 0, 0, 0)])
james.load()
james.show_skills(2)
