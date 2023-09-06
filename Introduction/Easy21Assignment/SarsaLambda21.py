import numpy as np
from collections import defaultdict

# The self-made environment for Easy21.
from Environment import Easy21Environment

# Import for plotting the state-value-function:
import matplotlib.pyplot as plt

# Importing pickle so i can load the action-value function from Monte-Carlo.
import pickle

class SarsaLambdaControl:
    
    def __init__(self, env, gamma=1.0, lamb = 0.5, N0 = 100):
        
        # We are using the Easy21-environment.
        self.env = env

        # In this task, we are setting the discount factor to 1.
        self.gamma = gamma

        # Setting up lambda to be the specified value:
        self.lamb = lamb
        
        # We are using a dynamic epsilon which changes during the episode,
        # in accordance to how many times we have visited that state and so forth.

        # This constant N0 is set to 100
        self.N0 = N0

        # Dictionaries for storing Q-values and the number of visits to specific states/ (state-action) pairs.
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(env.action_space.n))

    

    def learn_episode(self):
        """
        Generates an episode and updates the action-value function for each step taken.
        The updates are made according to the backward-view TD-Sarsa algorithm.
        """
        
        # Env.reset() simply returns the initial state.
        state = self.env.reset()
        action = self.get_action(state)
        
        # Initializing the eligibility trace:
        E = {}

        while True:

            # We see what happens after doing our first action:
            next_state, reward, done = self.env.step(action)

            # We sample our next action according to our epsilon-greedy policy:
            next_action = self.get_action(state)

            # Updating the eligibility trace:
            E = defaultdict(int, {key: value * self.gamma * self.lamb for key, value in E.items()})
            E[(state, action)] += 1
            self.N[state][action] += 1


            if done:
                # Some update to the action-value function.
                # Only the action-values in the eligibility trace should be updated.
                # The task says that we should use the same learning rate as in Monte Carlo: l/self.N[state][action]
                # Since state-action took us to terminal state with some reward for landing in that state,
                # We don't calculate the usual TD-error, as there is no next_action to take here.
                
                alpha = 1/self.N[state][action]
                td_error = reward - self.Q[state][action]

                for sa in E:
                    state, action = sa[0], sa[1]
                    self.Q[state][action] += alpha * td_error * E[sa]
                
                # Jumping out of the training episode.
                break
            
            # Some TD-update to the action-value function:
            # This is for the usual case where we are not in the terminal state yet,
            # so we update our action-value function based on usual TD-error.
            
            alpha = 1/self.N[state][action]
            td_error = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]

            for sa in E:
                    state, action = sa[0], sa[1]
                    self.Q[state][action] += alpha * td_error * E[sa]

            state, action = next_state, next_action

    

    
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



    def get_policy(self):
        """
        Returns the current policy based on the action-value function.
        """
        policy = defaultdict(int)
        for state in self.Q:
            policy[state] = np.argmax(self.Q[state])

        return policy

    def compute_mse(self, q_star):
        """
        Simple method which calculates the mean-square error between two action-value functions.
    
        Parameter:
            q_star (dict): The optimal action-value function, which in our case is obtained from Monte Carlo control.
        """
        return sum( (self.Q[state][action] - q_star[state][action])**2 for state in q_star for action in range(len(q_star[state] ))) / ( len(q_star) * self.env.action_space.n)

    def learn(self, num_episodes, q_star = None):
        """
        Executes a number of episodes of on-policy learning through sampling.
        Returns:
            1) The optimal policy as a dictionary with state and action.
            2) A list containing the MSE-values for each time step.
                - If q_star is not specified, then this list will be empty.
        """
        mse_values = []

        for i in range(1, num_episodes + 1):
            
            self.learn_episode()
            if not q_star == None:
                mse_values.append(self.compute_mse(q_star))

            if (i % 10_000 == 0):
                print(i)

        return self.get_policy(), mse_values


    def plot_error_lambda(mc_action_value, num_episodes = 1000):
        """
        Plotting the MSE for different values of lambda.
        The lambdas which are used are in range [0, 1] with step size 0.1
        """        
        mse_values = []
        for i in range(11):
            agent = SarsaLambdaControl(Easy21Environment(), lamb = i / 10)
            agent.learn(num_episodes)    
            MSE = sum( (agent.Q[state][action] - mc_action_value[state][action])**2 for state in mc_action_value for action in range(len(mc_action_value[state] ))) / ( len(mc_action_value) * agent.env.action_space.n)
            mse_values.append([i/10, MSE])
        
        fig = plt.figure(figsize = (12, 6))
        ax = fig.add_subplot(111)
        
        lambdas, mse_values = zip(*mse_values)
        ax.plot(lambdas, mse_values, marker='o')

        ax.set_xlabel("Lambda")
        ax.set_ylabel("MSE")
        ax.set_title("MSE for different values of lambda:")
        ax.grid(True)

        plt.show()


        
    
    def plot_state_value_function(self):

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

    def plot_mse(agent0, agent1, q_star, num_episodes = 1_000):
        
        _, mse0 = agent0.learn(num_episodes, q_star)
        _, mse1 = agent1.learn(num_episodes, q_star)

        fig = plt.figure(figsize = (12, 6))

        ax = fig.add_subplot(111)

        ax.plot(mse0, label = 'λ = 0')
        ax.plot(mse1, label = 'λ = 1')

        ax.set_xlabel('Episodes')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('MSE between SarsaLambda and Q*')

        ax.legend()

        plt.show()


# Retrieving the action-value function from the MonteCarlo-algorithm
# This with-open syntax does not create a new inner scope. Only method declarations, class definitions and modules have this property.
with open ('PracticalRL/Pickle/MonteCarloActionValue.pkl', 'rb') as f:
    mc_action_value = pickle.load(f)

### Plots the MSE after 1000 episodes for different values of lambda
# SarsaLambdaControl.plot_error_lambda(mc_action_value)

### For plotting the MSE of lambda = 0 and lambda = 1:
# agent0 = SarsaLambdaControl(Easy21Environment(), lamb = 0)
# agent1 = SarsaLambdaControl(Easy21Environment(), lamb = 1)
# SarsaLambdaControl.plot_mse(agent0, agent1, mc_action_value, num_episodes = 1_000)



"""
agent = SarsaLambdaControl(Easy21Environment())
policy, _ = agent.learn(1_000_000)

for i in range(11):
    agent = SarsaLambdaControl(Easy21Environment(), lamb = i / 10)
    agent.learn(1000)
    MSE = sum( (agent.Q[state][action] - mc_action_value[state][action])**2 for state in mc_action_value for action in range(len(mc_action_value[state] ))) / ( len(mc_action_value) * agent.env.action_space.n)

    with open ('Easy21Assignment/MSE.txt', 'a') as f:
        f.write(f'Lambda = {i / 10} : MSE =  {MSE} \n')

policy, _ = agent.learn(100_000)

with open('PracticalRL/Policies/SarsaLambda21.txt', 'w') as f:
        sorted_dict = dict(sorted(policy.items(), key = lambda x: (x[0][0], x[0][1])))
        for state in sorted_dict:
            f.write(str(state) + ": " + str(policy[state]) + "\n")
        f.close()

# Plot the state-value function
agent.plot_state_value_function()
"""


# agent = SarsaLambdaControl(Easy21Environment())
# policy, _ = agent.learn(100_000)

# with open('PracticalRL/Policies/SarsaLambda21.txt', 'w') as f:
#         sorted_dict = dict(sorted(policy.items(), key = lambda x: (x[0][0], x[0][1])))
#         for state in sorted_dict:
#             f.write(str(state) + ": " + str(policy[state]) + "\n")
#         f.close()
