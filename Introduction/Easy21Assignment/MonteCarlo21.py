import numpy as np
from collections import defaultdict

# The self-made environment for Easy21.
from Environment import Easy21Environment

# Import for plotting the state-value-function:
import matplotlib.pyplot as plt

# Importing pickle so i can save the action-value function.
import pickle

class MonteCarloControl:
    
    def __init__(self, env, discount_factor=1.0, N0 = 150):
        
        # We are using the Easy21-environment.
        self.env = env

        # In this task, we are setting the discount factor to 1.
        self.discount_factor = discount_factor
        
        # We are using a dynamic epsilon which changes during the episode,
        # in accordance to how many times we have visited that state and so forth.

        # This constant N0 is set to 100
        self.N0 = N0

        # Dictionaries for storing Q-values and the number of visits to specific states/ (state-action) pairs.
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(env.action_space.n))

    

    def generate_episode(self):
        """
        Generates an episode using current Q-values.
        """
        
        episode = []
        
        # Env.reset() simply returns the initial state.
        state = self.env.reset()
        
        while True:

            action = self.get_action(state)
            next_obs, reward, done = self.env.step(action)
            episode.append((state, action, reward))

            if done:
                break
            
            state = next_obs

        return episode
    
    def get_action(self, state):
        """
        Decides which action we should take in a state, according to our policy.
        Since we should use a dynamic epsilon in this task, the method gets more complicated than usual.
        """

        epsilon = self.N0 / (self.N0 + sum(self.N[state]))

        if (np.random.random() < epsilon):
            return np.random.randint(self.env.action_space.n)
        
        else:
            return np.argmax(self.Q[state])

    def update_q_values(self, episode):
        """
        Updates the action-value function estimate using the episode history.
        """

        G = 0
        visited_state_actions = set()

        for t in reversed(range(len(episode))):  # Loop through the episode
            
            state, action, reward = episode[t]
            G = reward + self.discount_factor * G

            if (state, action) not in visited_state_actions:
                
                visited_state_actions.add((state, action))
                self.N[state][action] += 1

                # Updating the action-value-function estimate, with the running average method.
                # µ_k = µ_(k-1) + 1/k * (x_k - µ_(k-1))
                self.Q[state][action] += (1 / self.N[state][action]) * (G - self.Q[state][action] )


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
        
        for i in range(1, num_episodes + 1):
            
            episode = self.generate_episode()
            self.update_q_values(episode)
            
            if (i % 10000 == 0):
                print(i)


        return self.get_policy(), dict(self.Q)

    
    def plot(self):

        # Extract the state-value function from the agent
        V = defaultdict(float)


        for state, actions in self.Q.items():
            V[state] = np.max(actions)

        # Prepare matrix to hold V values
        X = np.arange(1, 11) # Dealer showing
        Y = np.arange(1, 22) # Player sum
        Z = np.zeros((len(Y), len(X))) # State-value

        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                Z[j][i] = V[(x, y)]

        
        # This was just an experiment, so that i could understand the idea behind meshgrids.
        ## A, B = self.basic_meshgrid(X, Y)

        X, Y = np.meshgrid(X, Y)
        

        # Plotting
        fig = plt.figure(figsize=(12.19, 6.86))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('Dealer showing')
        ax.set_ylabel('Player sum')
        ax.set_zlabel('Value')
        ax.set_title('State-value function')
        ax.plot_surface(X, Y, Z, cmap='coolwarm')
        
        plt.show()


agent = MonteCarloControl(Easy21Environment())
policy, action_value_function = agent.learn(1_000_000)

# Saving the action-value-dictionary:
with open ('Pickle/MonteCarloActionValue.pkl', 'wb') as f:
    pickle.dump(action_value_function, f)


with open('Policies/MonteCarlo21.txt', 'w') as f:
        sorted_dict = dict(sorted(policy.items(), key = lambda x: (x[0][0], x[0][1])))
        for state in sorted_dict:
            f.write(str(state) + ": " + str(policy[state]) + "\n")
        f.close()

# Plot the state-value function
agent.plot()