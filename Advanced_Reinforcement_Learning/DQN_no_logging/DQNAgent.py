import torch

from Config.Agent import Agent
from Config.Config import Config as confg
from QNetwork import QNetwork
from Buffer import BasicBuffer
import gymnasium as gym
import numpy as np


class DQNAgent(Agent):
    
    class Config(confg):

        wandb_name = "DQN-CartPole"
        env = "CartPole-v1"
        ob_dim = 4
        ac_dim = 2
        hidden_dim = 600

        # Learning rate should not be a concern.
        lr = 0.001


        ### The epsilon value might need to be changed, a decay method might be needed.
        epsilon = 0.1
        flatline_episode = 3000
        
        # I don't believe that the value of the discount factor is all that important.
        gamma = 0.99


        # Not too sure about what these values should be.
        # Batch size, min_buffer_size, buffer_capacity and update_target_network_freq are the main culprits.
        batch_size = 6
        min_buffer_size = 10000
        buffer_capacity = 50000
        
        # Might need to look at how often this one should be updated.
        update_target_network_freq = 35

        episodes = 20_000
        eval_freq = 100
        

    def __init__(self, cfg):
        super().__init__(cfg)

        self.lr = cfg.lr
        self.epsilon = cfg.epsilon
        self.gamma = cfg.gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_values = QNetwork(self.cfg.ob_dim, self.cfg.ac_dim, self.cfg.hidden_dim).to(self.device)
        self.target_network = QNetwork(cfg.ob_dim, cfg.ac_dim, cfg.hidden_dim).to(self.device)
        self.update_target_network()

        # A mysterious 1 appears instead of ac dim, need to figure out what that is about.
        self.buffer = BasicBuffer.make_default(cfg.buffer_capacity, cfg.ob_dim, 1, wrap=True)

        self.optimizer = torch.optim.Adam(self.q_values.parameters(), cfg.lr)
        self.loss = torch.nn.MSELoss()

    def exploration_action(self):
        return np.random.randint(self.cfg.ac_dim)

    def greedy_action(self, state):
        return torch.argmax(self.q_values(torch.tensor(state).to(self.device))).item()

    def act(self, state):
        return self.exploration_action() if np.random.rand() < self.epsilon else self.greedy_action(state)

    def save(self, path):
        """Saving the model as a dictionary to minimize file size"""
        torch.save(self.q_values.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path); self.q_values.load_state_dict(state_dict); self.update_target_network()

    def store_transition(self, ob, ac, rew, next_ob, done):
        self.buffer << {'ob': [ob], 'ac': [ac], 'rew': [rew], 'next_ob': [next_ob], 'done': [done]}

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_values.state_dict())
    
    def epsilon_decay(self, i):
        """Using an epsilon decay function similar to the one used in the original DQN-paper."""
        end_point = 1 - self.epsilon; start_point = 1
        self.epsilon = start_point - (start_point - end_point) * (i / self.cfg.flatline_episode) if i < self.cfg.flatline_episode else end_point

    def update_q_values(self):

        if self.buffer.size < self.cfg.min_buffer_size:
            return None
        
        else:

            ob, ac, rew, next_ob, done = self.buffer.sample(batch_size = self.cfg.batch_size, device = self.device)

            # Next ob is a tensor [batch_size, ob_dim]
            # Dim1 specifies that we want to take the max value of each row
            # The returned tensors have shape (batch_size, ), representing the target_max for each sample in the batch
            # Target_max represents the maximum value of the next ob, given the action that maximizes the q value, while _ represents the index of the max value
            target_max, _ = self.target_network(next_ob).max(dim=1)

            # td_target is a tensor (batch_size, )
            # It tells us what the target value for each sample in the batch should be.
            td_target = rew + self.cfg.gamma * target_max * (1 - done)
            
            # Action values is a tensor (batch_size, )
            # It tells us what the predicted value for each sample in the batch is.
            # q_values(ob) returns a tensor (batch_size, ac_dim), where each row represents the q values for the given ob.
            # We use gather to get the q value for the action that was taken. ac.view(-1, 1) converts the array of actions into a column vector.
            ### Currently however, the actions are already a column vector, so this is not necessary.
            # The column vector containing action values stores indices used to get the q values for the actions that were taken.

            # The result is a tensor (batch_size, 1), which we squeeze to get a tensor (batch_size, )
            # This is the predicted action value q(s, a) for each sample in the batch.

            predicted_action_values = self.q_values(ob).gather(1, ac.view(-1, 1)).squeeze()
            
            # We use mean squared error to calculate the loss between the predicted action values and the target values.

            loss = self.loss(td_target, predicted_action_values)

            ### Backpropagation
            
            # Reset the gradients to zero
            self.optimizer.zero_grad()
            
            # Calculate the gradients
            loss.backward()

            # Update the weights, using the optimization algorithm Adaptive Moment Estimation (Adam).
            self.optimizer.step()
            
            return loss
    
    def play_cartpole(self, episodes = 10):
        env = gym.make(agent.cfg.env, render_mode = "human")
        for i in range(1, episodes + 1):
            episode_return = 0
            obs, _ = env.reset()
            while True:
                action = agent.greedy_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                obs = next_obs; episode_return += reward
                if terminated or truncated:
                    break
            print(f'Episode {i} return: {episode_return}')      

if __name__ == '__main__':
    agent = DQNAgent(DQNAgent.Config())
    print("Agent created.")
    agent.load("Advanced_Reinforcement_Learning\DQN_no_logging\models\soppDQN.pyt")
    agent.play_cartpole()
    # print(len(agent.buffer))
    # print(agent.buffer.data)
    # print("CUDA is available") if torch.cuda.is_available() else print("CUDA is not available")
