import numpy as np
import gymnasium as gym
from collections import deque

import torch
import torch.nn as nn
from QNetwork import QNetwork
from Connect_four import ConnectFour
from random import sample

class QLearningAgent():
    def __init__(self, action_dim, observed_dim, learning_rate_initial, epsilon, gamma, hidden_dim, decay_rate = 0.001, batch=200, maxlen=2000):
        self.action_dim = action_dim
        self.observed_dim = observed_dim
        self.learning_rate_initial = learning_rate_initial
        self.learning_rate = learning_rate_initial
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.gamma = gamma
        self.batch = batch
        
        self.this_episode = 0
        self.decay_rate = decay_rate
        
        self.Q_network = QNetwork(object_dim=observed_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.opponent_Q_network = QNetwork(object_dim=observed_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.buffer = deque(maxlen=maxlen)
        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), self.learning_rate)
        
        self.loss = nn.MSELoss()
    
    def decay_epsilon(self):
        self.epsilon = self.epsilon_initial * 1/(1 + self.decay_rate * self.this_episode)
        
    def act(self, value_func, env):
        if np.random.rand() < self.epsilon:
            legal_moves = [x for x in range(7) if env.is_valid_location(x)]
            return np.random.choice(legal_moves)
        else:
            legal_values = [value_func[x] for x in range(7) if env.is_valid_location(x)]
            legal_moves = [x for x in range(7) if env.is_valid_location(x)]
            
            index = torch.argmax(torch.Tensor(legal_values)).item()
            return legal_moves[index]

        
    def save(self, filename):
        torch.save(self.Q_network.state_dict(), filename)

    def load(self, filename):
        self.Q_network.load_state_dict(torch.load(filename))
        #self.Q_network.eval() #Tar av dropout
        
    def copy_nn(self):
        self.opponent_Q_network.load_state_dict(self.Q_network.state_dict())
        #self.Q_network.eval() #Tar av dropout
    
    def compute_loss(self, batch):
        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute current Q-values
        current_q_values = self.Q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute next Q-values
        next_q_values = self.Q_network(next_states).max(1)[0]
        
        # Compute target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())

        return loss
  
    def train(self, episodes, render = False):
        #Delete this maybe?
        state = torch.tensor([0])
        for i in range(episodes):
            self.this_episode += 1
            
            # Spiller mot seg selv
            if i % 100 == 0:
                self.copy_nn()
            
            env = ConnectFour()

            observation = env.reset()
            state = torch.tensor(observation, dtype=torch.float32)

            self.decay_epsilon()
            
            score = 0
            
            while(True):
                # Player 1
                value_func = self.Q_network.forward(state) # Predict
                action = self.act(value_func, env) # Get action
                
                observation, reward, done = env.step(action) # Do action
                next_state  = torch.tensor(observation, dtype=torch.float32)
                score += reward
                self.buffer.append((state, action, reward, next_state, done))
                
                state = next_state
                
                if(done):
                    break
                
                # Training
                if len(self.buffer) > self.batch:
                    random_sample = sample(self.buffer, self.batch)
                    
                    states = torch.stack([x[0].to(torch.float) for x in random_sample])
                    actions = torch.tensor([x[1] for x in random_sample], dtype=torch.float32)
                    rewards = torch.tensor([x[2] for x in random_sample], dtype=torch.float32)
                    next_states = torch.stack([x[3].to(torch.float) for x in random_sample])
                    dones = torch.tensor([x[4] for x in random_sample], dtype=torch.float32)
                    
                    target_max, _ = self.Q_network(next_states).max(dim=1)
                    
                    td_target = rewards + self.gamma * target_max * (1 - dones)
                    
                    predicted_action_values = self.Q_network(states).gather(1, actions.view(-1, 1).long()).squeeze()
                    
                    loss = self.loss(td_target, predicted_action_values)
            
                    # optimize the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Player 2
                value_func = self.opponent_Q_network.forward(-state) # Predict
                action = self.act(value_func, env) # Get action
                
                observation, reward, done = env.step(action) # Do action
                reward = -reward # If opponent wins, reward is negative
                score += reward
                next_state  = torch.tensor(observation, dtype=torch.float32)
                
                self.buffer.append((state, action, reward, next_state, done))
                
                state = next_state
                if(done):
                    break
                
            if self.this_episode % 100 == 0:
                print("episode: ", self.this_episode, "score: ", score, "epsilon: ", self.epsilon)
                print("State: ", state.reshape(6,7))
            
if __name__ == '__main__':
    agent = QLearningAgent(
        7, 42, # action_dim, observed_dim
        learning_rate_initial=0.0001, 
        epsilon=0.5, 
        gamma=0.99,
        hidden_dim=100, 
        decay_rate=0.0005
        )
    agent.load("connect4.pk1")
    try:
        agent.train(episodes=1_000_000)
    except KeyboardInterrupt:
        print("\nSaving!")
        agent.save("connect4.pk1")
        pass
    
    agent.save("connect4.pk1")
