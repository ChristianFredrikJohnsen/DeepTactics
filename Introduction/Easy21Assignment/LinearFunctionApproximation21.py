import numpy as np
from collections import defaultdict

# The self-made environment for Easy21.
from Environment import Easy21Environment

# Importing function which provides cartesian product:
from itertools import product

# Import for plotting the state-value-function:
import matplotlib.pyplot as plt

# Importing pickle so i can load the action-value function from Monte-Carlo.
import pickle

class LinearFunctionApproximationControl:
    
    def __init__(self, env, gamma = 1.0, lamb = 0.5, epsilon = 0.05, alpha = 0.01):
        
        # We are using the Easy21-environment.
        self.env = env

        # In this task, we are setting the discount factor to 1.
        self.gamma = gamma

        # Setting up lambda to be the specified value:
        self.lamb = lamb
        
        # Epsilon and alpha are set to be constant:
        self.epsilon = epsilon
        self.alpha = alpha

        # Intializing some weights, One weight for each feature:
        self.w = np.zeros(36)
    

    def learn_episode(self):
        """
        Generates an episode and updates the action-value function for each step taken.
        The updates are made according to the backward-view TD-Sarsa algorithm.
        """
        
        # Env.reset() simply returns the initial state.
        state = self.env.reset()
        action = self.get_action(state)
        
        # Getting the feature vector representation of our state-action pair:
        feature_vector = self.get_feature_vector(state, action)

        # Initializing the eligibility trace:
        E = np.zeros(36)

        while True:
            
            # We take a step in the environment and see what happens. 
            next_state, reward, done = self.env.step(action); 

            # We sample our next action according to our epsilon-greedy policy:
            next_action = self.get_action(state)

            # We must remember to represent the next state, next action pair as a feature vector.
            next_feature_vector = self.get_feature_vector(next_state, next_action)

            # Updating the eligibility trace:
            E *= self.lamb * self.gamma
            E += feature_vector
            

            if done:
                
                # Some update to the action-value function.
                # Only the action-values in the eligibility trace should be updated.
                # We don't calculate the usual TD-error, as there is no next_action to take here.

                # The dot product between feature vector and the weights essentially gives us the approximated q-value.
                self.w += self.alpha * (reward - np.dot(feature_vector, self.w)) * E

                # Jumping out of the training episode.
                break
            
            # Some TD-update to the action-value function:
            # This is for the usual case where we are not in the terminal state yet,
            # so we update our action-value function based on usual TD-error.
            self.w += self.alpha * (reward + np.dot(next_feature_vector, self.w) - np.dot(feature_vector, self.w)) * E

            # Setting up for next action:
            state, action, feature_vector = next_state, next_action, next_feature_vector


    def get_feature_vector(self, state, action):
        """
        Takes in a state-pair of the form (dealer_sum, player_sum, action).
        and returns a 1D vector with 36 entries, where each entry corresponds to some specific feature.
        """

        # Setting up the feature vector:
        feature_vector = np.zeros(36)

        # Extracting the useful information for the features:
        dealer_sum = state[0]
        player_sum = state[1]
        action = "stick" if action == 0 else "hit"

        # Defining the features:
        dealer_ranges = [(1, 4), (4, 7), (7, 10)]
        player_ranges = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]
        actions = ["hit", "stick"]

        for i, (d_range, p_range, a) in enumerate(product(dealer_ranges, player_ranges, actions)):
            if d_range[0] <= dealer_sum <= d_range[1] and p_range[0] <= player_sum <= p_range[1] and action == a:
                feature_vector[i] = 1
        
        return feature_vector

    
    def get_action(self, state):
        """
        Decides which action we should take in a state, according to our policy.
        Our policy is epsilon-greedy with a constant epsilon. 
        Standard value for epsilon is 0.05
        """

        if (np.random.random() < self.epsilon):
            return np.random.randint(self.env.action_space.n)
        
        else:
            return np.argmax([np.dot(self.get_feature_vector((state[0], state[1]), a), self.w) for a in range(self.env.action_space.n)])

    def get_policy(self):
        """
        Returns the current policy based on the action-value function.
        """
        policy = defaultdict(int)
        dealer = [i for i in range(1, 11)]
        player = [i for i in range(1, 22)]
        states = product(dealer, player)

        for state in states:
            policy[state] = np.argmax([np.dot(self.get_feature_vector((state[0], state[1]), a), self.w) for a in range(self.env.action_space.n)])

        return policy
    
    def get_action_values(self):
        """
        Returns the action values which the agent has found:
        """
        q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        dealer = [i for i in range(1, 11)]
        player = [i for i in range(1, 22)]
        states = product(dealer, player)

        for state in states:
            for action in range(self.env.action_space.n):
                q[state][action] = np.dot(self.get_feature_vector((state[0], state[1]), action), self.w)

        return q

    def learn(self, num_episodes):
        """
        Executes a number of episodes of on-policy learning through sampling.
        Returns:
            1) The optimal policy as a dictionary with state and action.
        """
        for i in range(1, num_episodes + 1):
            
            self.learn_episode()

            if (i % 10_000 == 0):
                print(i)
        
        return self.get_policy()
       
    def learn_with_mse(self, num_episodes, q_star):
        """
        Executes a number of episodes of on-policy learning through sampling.
        Returns:
            1) The optimal policy as a dictionary with state and action.
            2) A list containing the MSE-values for each time step.
        """
        mse_values = []

        for i in range(1, num_episodes + 1):
            
            self.learn_episode()
            mse_values.append(LinearFunctionApproximationControl.compute_mse(self, q_star))

            if (i % 10_000 == 0):
                print(i)
        
        return self.get_policy(), mse_values

    def plot_mse(q_star, num_episodes = 1_000):
        
        # Setting up the agents
        agent0 = LinearFunctionApproximationControl(Easy21Environment(), lamb = 0)
        agent1 = LinearFunctionApproximationControl(Easy21Environment(), lamb = 1)

        # Finding the list of calculated MSE for each episode
        _, mse0 = agent0.learn_with_mse(num_episodes, q_star)
        _, mse1 = agent1.learn_with_mse(num_episodes, q_star)

        fig = plt.figure(figsize = (12, 6))

        ax = fig.add_subplot(111)

        ax.plot(mse0, label = 'λ = 0')
        ax.plot(mse1, label = 'λ = 1')

        ax.set_xlabel('Episodes')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('MSE between Linear Function Approximation and Q*')

        ax.legend()

        plt.show()

    def compute_mse(agent, q_star):
        """
        Simple method which calculates the mean-square error between two action-value functions.
    
        Parameter:
            q_star (dict): The optimal action-value function, which in our case is obtained from Monte Carlo control.
        """
        return sum( (np.dot(agent.get_feature_vector(state, action), agent.w) - q_star[state][action])**2 for state in q_star for action in range(len(q_star[state] ))) / ( len(q_star) * agent.env.action_space.n)

    def plot_error_lambda(mc_action_value, num_episodes = 1000):
        """
        Plotting the MSE for different values of lambda.
        The lambdas which are used are in range [0, 1] with step size 0.1
        """        
        mse_values = []
        for i in range(11):
            agent = LinearFunctionApproximationControl(Easy21Environment(), lamb = i / 10)
            policy = agent.learn(num_episodes)    
            MSE = sum( (np.dot(agent.get_feature_vector(state, policy[state]), agent.w) - mc_action_value[state][action])**2 for state in mc_action_value for action in range(len(mc_action_value[state])) ) / ( len(mc_action_value) * agent.env.action_space.n)
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


with open("PracticalRL\Pickle\MonteCarloActionValue.pkl", "rb") as f:
    mc_action_value = pickle.load(f)


# LinearFunctionApproximationControl.plot_error_lambda(mc_action_value, num_episodes = 10_000)

# agent = LinearFunctionApproximationControl(Easy21Environment())
# policy = agent.learn(200_000)

# with open('PracticalRL/Policies/LinearApproximation21.txt', 'w') as f:
#         sorted_dict = dict(sorted(policy.items(), key = lambda x: (x[0][0], x[0][1])))
#         for state in sorted_dict:
#             f.write(str(state) + ": " + str(policy[state]) + "\n")
#         f.close()

LinearFunctionApproximationControl.plot_mse(mc_action_value, num_episodes = 3_000)
# LinearFunctionApproximationControl.plot_error_lambda(mc_action_value, num_episodes = 10_000)

"""
agent_bad = LinearFunctionApproximationControl(Easy21Environment())
policy = agent_bad.learn(100)
q_bad = agent_bad.get_action_values()

agent_better = LinearFunctionApproximationControl(Easy21Environment())
agent_better.learn(10_000)
q_better = agent_better.get_action_values()


with open('PracticalRL/Policies/bad.txt', 'w') as f:
        sorted_dict = dict(sorted(q_better.items(), key = lambda x: (x[0][0], x[0][1])))
        for state in sorted_dict:
            f.write(f'{(state[0], state[1], 0)}: {q_bad[state][0]} \n')
        for state in sorted_dict:
            f.write(f'{(state[0], state[1], 1)}: {q_bad[state][1]} \n')
        f.close()

with open('PracticalRL/Policies/badPolicy.txt', 'w') as f:
        sorted_dict = dict(sorted(policy.items(), key = lambda x: (x[0][0], x[0][1])))
        for state in sorted_dict:
            f.write(str(state) + ": " + str(policy[state]) + "\n")
        f.close()

with open('PracticalRL/Policies/better.txt', 'w') as f:
        sorted_dict = dict(sorted(q_better.items(), key = lambda x: (x[0][0], x[0][1])))
        for state in sorted_dict:
            for action in range(agent_better.env.action_space.n):
                f.write(f'{(state[0], state[1], action)}: {q_better[state][action]} \n')
        f.close()
"""
