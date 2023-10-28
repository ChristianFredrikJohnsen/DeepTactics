import numpy as np; from collections import deque; import torch
import torch.nn as nn; from QNetwork import QNetwork
from env import ConnectFourEnvironment; from debug_utils.print_board import print_status
from icecream import ic; from random import sample; from collections import namedtuple

class QLearningAgent():
    """
    DQN Agent for Connect 4.
    """
    
    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done')) # A named tuple to store state-action transitions.
    
    
    def __init__(self, action_dim, observed_dim, learning_rate_initial, epsilon, gamma, hidden_dim, decay_rate = 0.001, batch=32, min_batch_size = 3000, maxlen=1_000_000, update_target_network_freq=200):
        """
        Setting up all the parameters for the agent.
        Intializing the Q-network, the target network, the opponent network, the replay buffer, the loss function and the optimizer.
        Also setting up the device, which is either the GPU or the CPU.
        """

        self.action_dim = action_dim; self.observed_dim = observed_dim; self.learning_rate_initial = learning_rate_initial
        self.learning_rate = learning_rate_initial; self.epsilon = epsilon; self.epsilon_initial = epsilon; self.min_batch_size = min_batch_size
        self.gamma = gamma; self.batch = batch; self.decay_rate = decay_rate; self.update_target_network_freq = update_target_network_freq
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Setting up CUDA if available.

        self.Q_network = QNetwork(object_dim=observed_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_network = QNetwork(object_dim=observed_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.opponent_Q_network = QNetwork(object_dim=observed_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.buffer = deque(maxlen=maxlen)

        self.loss = nn.MSELoss() # Calculate how bad the network is.
        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), self.learning_rate) # Calculate how to make the network less bad, based on loss.
    
    def decay_epsilon(self, episode_num, min_rate = 0.03):
        """
        Decay the epsilon value based on epsiode number.
        """
        self.epsilon = max(self.epsilon_initial / (1 + self.decay_rate * episode_num), min_rate) # Decay epsilon, always have a minimum exploration rate of 3%.
        
    def act(self, state):
        """
        We are implementing a basic epsilon-greedy policy.
        """

        if np.random.rand() < self.epsilon:
            return np.random.randint(7)
        
        else: # Do greedy action / "best" action
            qvals = self.Q_network(state)
            return torch.argmax(qvals).item()

    def save(self, filename):
        """
        Saves the model parameters to a file.
        """
        torch.save(self.Q_network.state_dict(), filename)
        torch.save(self.opponent_Q_network.state_dict(), filename + "_opponent")

    def load(self, filename):
        """
        Loads the model parameters from a file.
        """
        self.Q_network.load_state_dict(torch.load(filename, map_location=self.device))
        self.opponent_Q_network.load_state_dict(torch.load(filename + "_opponent", map_location=self.device))
        self.target_network.load_state_dict(self.Q_network.state_dict())
        
    def copy_nn(self):
        """
        Copy the current Q-network to the opponent Q-network.
        We must clear the replay buffer when copying the network, to avoid using redundant data.
        To provide a practical example, there might have been a strategy which was good in the beginning of the training, 
        but which the new agent has learned to counter.
        """
        self.opponent_Q_network.load_state_dict(self.Q_network.state_dict())
        self.buffer.clear() # Clear the replay buffer when copying the network.
    
    def compute_loss(self, batch, print = False):
        """
        Compute the loss for a batch of state-action transitions.
        We are currently using the MSE loss function.
        """
        states, actions, rewards, next_states, dones = batch # Unpack batch        
        if print:
            ic(states, actions, rewards, next_states, dones)

        current_q_values = self.Q_network(states).gather(1, actions).squeeze() # Compute current Q-values using policy network
        if print:
            ic(current_q_values)

        next_q_values = self.target_network(next_states).max(1)[0] # Compute next Q-values using target network
        if print:
            ic(next_q_values)

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values # Compute target Q-values
        if print:
            ic(target_q_values)

        loss = self.loss(current_q_values, target_q_values) # Compute loss
        if print:
            ic(loss)

        return loss
    
    def backprop(self, loss):
        """
        Do backpropagation to update the weights of the neural network.
        Pytorch does all the heavy lifting for us.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes, win_check = 250, render = False):
        """
        This is the main training loop.
        The agent plays connect 4 against itself, and trains the network based on samples from the replay buffer.
        The opponent is using a greedy policy.
        If the agent wins 90% of the games in the last 100 episodes, the opponent network is set to be equal to the agent network.
        This way, the agent is always playing against a strong opponent.
        """

        env = ConnectFourEnvironment(self.opponent_Q_network); results = np.zeros(win_check) # Specifying the environment, and setting up a list to store the results of the last 100 episodes.
        for episode_num in range(1, episodes + 1):
            
            if self.log_and_copy_network(episode_num, results, win_check): # If the opponent network was updated, we must update the environment such that the opponent is using the new network.
                env = ConnectFourEnvironment(self.opponent_Q_network)
            
            reward, final_state = self.play_until_done(env, episode_num)
            win = 1 if reward == 1 else 0 # Record if the agent won or not.
            results[episode_num % win_check] = win
            
            self.decay_epsilon(episode_num)

            # Update the target network.
            if episode_num % self.update_target_network_freq == 0:
                self.target_network.load_state_dict(self.Q_network.state_dict())

            if episode_num % 100 == 0:
                print_status(reward, episode_num, win_check, final_state, results, self.epsilon)
        

    def play_until_done(self, env, episode_num):
        """
        This is the where the network plays a game of connect 4 against itself.
        On odd-numbered episodes, the opponent is playing first, and on even-numbered episodes, the agent is playing second.
        The network is trained based on samples from the replay buffer.
        Returns: The reward and the final state of the game.
        """
        agent_moves_first = episode_num % 2 == 1 # If the episode number is odd, the agent is playing first.
        state = env.reset(agent_moves_first).to(self.device) # Reset the environment and get the initial state.
        
        while(True):
            state, reward, done = self.perform_action(env, state) # Perform an action in the environment, and add the state-action transition to the replay buffer.
            self.update_parameters(print = False) # Sample a batch from the replay buffer and train the network.
            if(done):
                return (reward, state) # Return the reward and the final state of the game.
        

    def perform_action(self, env, state, opponent=False):
        """
        Perform an action in the environment, and add the state-action transition to the replay buffer.
        Returns the next state, the reward, and whether the game is over.
        """
        action = self.opponent_act(state) if opponent else self.act(state)
        next_state, reward, done = env.step(action)
        
        # Add the state-action transition to the replay buffer, if it is not the opponent who is playing.
        if not opponent:
            self.buffer.append(QLearningAgent.Transition(state, action, reward, next_state, done))

        return next_state.to(self.device), reward, done
    
    def update_parameters(self, print):
        """
        Train the network by sampling a batch from the replay buffer.
        We are using a batch size of 200, and we are using MSE loss and Adam optimizer to train the network.
        """
        if len(self.buffer) > self.min_batch_size:
                    loss = self.compute_loss(self.get_random_samples(), print)
                    self.backprop(loss)

    def log_and_copy_network(self, episode_num, results, win_check):
        """
        Log the results of the last 100 episodes, and copy the network if the winrate is above 75%.
        Returns True if the opponent network was updated, and False otherwise.
        """
        if episode_num % win_check == 0:
            if np.mean(results) >= 0.9:
                print("Copying network!")
                self.copy_nn()
                return True
        return False

    def get_random_samples(self):
        """
        Extract a batch of random state-action transitions from the replay buffer.
        """
        random_sample = sample(self.buffer, self.batch)
        states = torch.stack([x.state for x in random_sample]).to(self.device)
        actions = torch.tensor([x.action for x in random_sample], dtype=torch.int64).unsqueeze(-1).to(self.device) # Making sure that the actions are of type long, and that the tensor has the correct shape.
        rewards = torch.tensor([x.reward for x in random_sample], dtype=torch.float32).to(self.device)
        next_states = torch.stack([x.next_state for x in random_sample]).to(self.device)
        dones = torch.tensor([x.done for x in random_sample], dtype=torch.float32).to(self.device)
        return (states, actions, rewards, next_states, dones)
    

if __name__ == '__main__':

    filename = "models/connect4_christian_abomination.pk1" # Get the parameters you are working with.
    
    # Intialize the agent.
    agent = QLearningAgent(
        action_dim=7, 
        observed_dim=42,
        learning_rate_initial=0.0001, 
        epsilon=0.1, 
        gamma=1, 
        hidden_dim=500, 
        decay_rate=0.001 
        )
    
    print(agent.device)
    # agent.load(filename) # Load the already trained agent
    
    # Start training. If you want to stop training, press ctrl + c, and the agent will be saved.
    try:
        agent.train(episodes=10_000_000)
        print("\nSaving!")
        agent.save(filename) # Save the agent after training. 
    except KeyboardInterrupt:
        print("\nSaving!")
        agent.save(filename) # Save the agent if training is interrupted. 
