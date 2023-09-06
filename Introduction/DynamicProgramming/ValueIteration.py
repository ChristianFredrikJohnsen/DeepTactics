import numpy as np

def value_iteration(states, actions, transition_probabilities, reward_function, discount_factor, theta=0.0001):
    
    V = np.zeros(len(states))  # initialize value-function
    
    while True:
        
        delta = 0
        
        # We want to evaluate the expected future reward of each state.
        for s in states:

            # We save the old approximation for state-value function for this state.
            v = V[s]

            # This part evaluates the new value-function for this state.
            # Basically, you find the maximum expected future reward, by brute-forcing through all of the actions, calculating the immediate reward, and then
            # calculating the expected future reward, which is done very briefly by looking at all the possible state-transitions and finding the value you have assigned to
            # those new states. You multiply probability by your expected reward from that state (your expected future reward v(s) is only an approximation at this point). 

            V[s] = max(sum(transition_probabilities[s][a][s_prime] * (reward_function[s][a][s_prime] + discount_factor * V[s_prime]) for s_prime in states) for a in actions)
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    return V



def value_iteration_step(states, actions, reward, transition_probabilities, discount, V_k):
    
    V_k_plus_1 = np.zeros(len(states))  # Initialize new value function

    for s in states:  # For each state
        value_per_action = []  # This list will hold the values for each action
        for a in actions:  # For each action
            value_for_a = 0  # Initialize the value for action 'a'
            for s_prime in states:  # For each next state
                # Calculate the sum of the reward and discounted future value, weighted by the transition probability
                value_for_a += transition_probabilities[s][a][s_prime] * (reward[s][a][s_prime] + discount * V_k[s_prime])
            value_per_action.append(value_for_a)  # Add the calculated value for action 'a' to the list
        # Select the maximum over the actions
        V_k_plus_1[s] = max(value_per_action)

    return V_k_plus_1












