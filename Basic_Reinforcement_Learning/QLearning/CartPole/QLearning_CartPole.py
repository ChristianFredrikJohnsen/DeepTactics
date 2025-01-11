from collections import defaultdict
import pickle
from Config.Config import Config as Confg
import numpy as np
import wandb
from Config.Agent import Agent

class QLearningAgent(Agent):

    class Config(Confg):

        # Env variables
        env = "CartPole-v1"
        ac_dim = 2
        ob_dim = 4

        # Hyperparameters
        lr = 0.3
        epsilon = 0.1
        gamma = 0.99

        # Training runs
        wandb_name = env
        episodes = 20_000
        eval_frequency = 1000



    def __init__(self, cfg, lr_decay = lambda lr, i: lr, epsilon_decay=lambda epsilon, i: epsilon):
        super().__init__(cfg)

        self.lr = cfg.lr
        self.lr_decay = lr_decay

        self.epsilon = cfg.epsilon
        self.epsilon_decay = epsilon_decay

        self.q_values = defaultdict(lambda: [0] * self.cfg.ac_dim)
        self.num_updates = 0

    def _greedy_action(self, state):
        return np.argmax(self.q_values[state])
    
    def _exploration_action(self):
        return np.random.randint(self.cfg.ac_dim)
    
    def act(self, state):
        if np.random.rand() < self.cfg.epsilon:
            return self._exploration_action()
        else:
            return self._greedy_action(state)
        
    def save(self, path):
        values = dict(self.q_values)
        with open(path, 'wb') as f:
            pickle.dump(values, f)
        
        # np.save(path, np.array(self.q_values))
    
    def load(self, path):

        with open(path, 'rb') as f:
            values = pickle.load(f)
        self.q_values = defaultdict(lambda: [0] * self.cfg.ac_dim)
        self.q_values.update(values)

    def update_q_values(self, state, reward, action, next_state):
        self.q_values[state][action] += self.cfg.lr * (reward + self.cfg.gamma * np.max(self.q_values[next_state])- self.q_values[state][action])

        self.num_updates += 1

    def decay_lr(self, i):
        self.lr = self.lr_decay(self.lr, i)

    def decay_epsilon(self, i):
        self.epsilon = self.epsilon_decay(self.epsilon, i)



if __name__ == '__main__':
    agent = QLearningAgent(QLearningAgent.Config())

    agent.q_values[(0, 0, 0, 0)] = [3, -4]
    print(agent.q_values[0, 0, 0, 0])

