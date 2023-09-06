import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

with open ('PracticalRL/Pickle/MonteCarloActionValue.pkl', 'rb') as f:
    Q = pickle.load(f)

def plot():

    # Extract the state-value function from the agent
    V = defaultdict(float)


    for state, actions in Q.items():
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

plot()