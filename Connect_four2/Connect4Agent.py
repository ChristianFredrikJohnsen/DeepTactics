import numpy as np
from collections import deque

import torch
import torch.nn as nn
from QNetwork import QNetwork
from Connect_four import ConnectFour
from icecream import ic

from random import sample

from colorama import Fore, Style, init


class QLearningAgent():
    def __init__(self, action_dim, observed_dim, learning_rate_initial, epsilon, gamma, hidden_dim, decay_rate = 0.001, batch=200, maxlen=2000):
        
        self.action_dim = action_dim; self.observed_dim = observed_dim; self.learning_rate_initial = learning_rate_initial
        self.learning_rate = learning_rate_initial; self.epsilon = epsilon; self.epsilon_initial = epsilon
        self.gamma = gamma; self.batch = batch
        
        self.this_episode = 0
        self.decay_rate = decay_rate
        
        self.Q_network = QNetwork(object_dim=observed_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.opponent_Q_network = QNetwork(object_dim=observed_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.buffer = deque(maxlen=maxlen)
        
        self.loss = nn.MSELoss() # Calculate how bad the network is.
        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), self.learning_rate) # Calculate how to make the network less bad, based on loss.
        
        init(autoreset=True) # Colorama lol

    
    def decay_epsilon(self):
        self.epsilon = self.epsilon_initial / (1 + self.decay_rate * self.this_episode)
        
    def act(self, value_func, env):
        if np.random.rand() < self.epsilon: # Do random action
            legal_moves = [x for x in range(7) if env.is_valid_location(x)]
            return np.random.choice(legal_moves)
        else: # Do greedy action / "best" action
            legal_values = [value_func[x] for x in range(7) if env.is_valid_location(x)]
            legal_moves = [x for x in range(7) if env.is_valid_location(x)]
            
            index = torch.argmax(torch.Tensor(legal_values)).item()
            return legal_moves[index]

        
    def save(self, filename):
        torch.save(self.Q_network.state_dict(), filename)

    def load(self, filename):
        self.Q_network.load_state_dict(torch.load(filename))
        
    def copy_nn(self):
        self.opponent_Q_network.load_state_dict(self.Q_network.state_dict())
    
    def compute_loss(self, batch):
        # Unpack the batch
        states, actions, rewards, next_states, dones = batch
        
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
        results = np.zeros(100)
        for episode_num in range(1, episodes + 1):
            self.this_episode = episode_num
            
            # Spiller mot seg selv
            if episode_num % 100 == 0 and np.mean(results) >= 0.75:
                print("Copying network!")
                print("Winrate last 100 episodes: ", np.mean(results))
                print("results: ", results)
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
                
                next_state, reward, done = env.step(action) # Do action
                next_state  = torch.tensor(next_state, dtype=torch.float32)
                score += reward
                self.buffer.append((state, action, reward, next_state, done))
                
                state = next_state
                
                if(done):
                    results[episode_num % 100] = reward if reward == 1 else 0
                    break
                
                # Training
                if len(self.buffer) > self.batch:
                    self.compute_loss(self.get_random_samples())
                    states, actions, rewards, next_states, dones = self.get_random_samples() # Get a bunch of samples.
                    
                    target_max, _ = self.Q_network(next_states).max(dim=1) # Calcluate the max_values for next state
                    
                    td_target = rewards + self.gamma * target_max * (1 - dones) # Calculate the td_target
                    
                    predicted_action_values = self.Q_network(states).gather(1, actions.view(-1, 1).long()).squeeze() # Calculate the max from the current states. 
                    
                    loss = self.loss(td_target, predicted_action_values) # Calculate the loss
            
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
                    results[self.this_episode % 100] = 0
                    break
                
            if self.this_episode % 100 == 0:
                
                if(score == 1):
                    print(f"{Fore.RED}episode: {self.this_episode} score: {score} epsilon: {self.epsilon} winrate: {np.mean(results)}{Style.RESET_ALL}")
                elif(score == -1):
                    print(f"{Fore.YELLOW}episode: {self.this_episode} score: {score} epsilon: {self.epsilon} winrate: {np.mean(results)}{Style.RESET_ALL}")

                #print("State:\n", str(torch.flip(state.reshape(6,7), [0]).numpy())[1:-1], "\n")
                self.print_board(torch.flip(state.reshape(6,7), [0]).numpy())
            
    def print_board(self, board):
        for i in range(6):
            for j in range(7):
                self.print_color(board[i][j])
            print()
    
    def print_color(self, num):
        if num == 1:
            print(f"{Fore.RED} 1 ", end='')     # Yellow
        elif num == -1:
            print(f"{Fore.YELLOW}-1 ", end='')  # Green
        elif num == 0:
            print(f"{Fore.WHITE} 0 ", end='')   # White

    
    def get_random_samples(self):
        random_sample = sample(self.buffer, self.batch)
        states = torch.stack([x[0].to(torch.float) for x in random_sample])
        actions = torch.tensor([x[1] for x in random_sample], dtype=torch.int64)
        rewards = torch.tensor([x[2] for x in random_sample], dtype=torch.float32)
        next_states = torch.stack([x[3].to(torch.float) for x in random_sample])
        dones = torch.tensor([x[4] for x in random_sample], dtype=torch.float32)
        return (states, actions, rewards, next_states, dones)
    

if __name__ == '__main__':

    # Get the parameters you are working with.
    filename = "models/connect4_dim=500.pk1"
    
    # Intialize the agent.
    agent = QLearningAgent(
        7, 42, # action_dim, observed_dim 
        learning_rate_initial=0.0001, 
        epsilon=0.5, 
        gamma=0.99, 
        hidden_dim=500, 
        decay_rate=0.001 
        )
    
    # Load the already trained agent
    agent.load(filename)
    
    # Start training. If you want to stop training, press ctrl + c, and the agent will be saved.
    try:
        agent.train(episodes=201)
    except KeyboardInterrupt:
        print("\nSaving!")
        agent.save(filename)

    # Save the agent after training.    
    agent.save(filename)
