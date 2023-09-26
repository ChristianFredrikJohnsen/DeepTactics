import numpy.random
import torch

from Config.Agent import Agent
from Config.Config import Config as confg
from QNetwork import QNetwork
from Buffer import BasicBuffer
import numpy as np


class DQNAgent(Agent):
    
    class Config(confg):

        wandb_name = "DQN-CartPole"
        env = "CartPole-v1"
        ob_dim = 4
        ac_dim = 2
        hidden_dim = 25

        lr = 0.001
        epsilon = 0.05
        gamma = 0.99
        batch_size = 100

        min_buffer_size = 200
        buffer_capacity = 1_000
        episodes = 2000
        eval_freq = 500
        update_target_network_freq = 100

    def __init__(self, cfg):
        super().__init__(cfg)

        self.lr = cfg.lr
        self.epsilon = cfg.epsilon
        self.gamma = cfg.gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_values = QNetwork(self.cfg.ob_dim, self.cfg.ac_dim, self.cfg.hidden_dim).to(self.device)
        self.target_network = QNetwork(cfg.ob_dim, cfg.ac_dim, cfg.hidden_dim).to(self.device)
        self.update_target_network()

        # A mysterious -1 in ac dim appears, need to figure out what that is about.
        self.buffer = BasicBuffer.make_default(cfg.buffer_capacity, cfg.ob_dim, cfg.ac_dim - 1, wrap=True)

        self.optimizer = torch.optim.Adam(self.q_values.parameters(), cfg.lr)
        self.loss = torch.nn.MSELoss()

    def exploration_action(self):
        return numpy.random.randint(self.cfg.ac_dim)

    def greedy_action(self, state):
        return torch.argmax(self.q_values(torch.tensor(state).to(self.device))).item()

    def act(self, state):
        return self.exploration_action() if np.random.rand() < self.epsilon else self.greedy_action(state)

    def save(self, path):
        torch.save(self.q_values.state_dict(), path)

    def load(self, path):
        self.q_values = torch.load(path); self.update_target_network()

    def store_transition(self, ob, ac, rew, next_ob, done):
        self.buffer << {'ob': [ob],
                        'ac': [ac],
                        'rew': [rew],
                        'next_ob': [next_ob],
                        'done': [done]
                        }

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_values.state_dict())

    def update_q_values(self):
        
        if self.buffer.size < self.cfg.min_buffer_size:
            return None
        
        else:
            
            ob, ac, rew, next_ob, done = self.buffer.sample(self.cfg.batch_size)
            ac = ac.type(torch.long)
            done = done.type(torch.int)

            target_max, _ = self.target_network(next_ob).max(dim=1)
            td_target = rew + self.cfg.gamma * target_max * (1 - done)

            old_values = self.q_values(ob).gather(1, ac.view(-1, 1)).squeeze()

            loss = self.loss(td_target, old_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss

if __name__ == '__main__':
    agent = DQNAgent(DQNAgent.Config())
    print('done')