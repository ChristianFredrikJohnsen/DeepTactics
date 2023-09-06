import numpy as np


def policy_evaluation_k(policy, states, transition_probabilities, reward_function, discount_factor, k):
    
    V = np.zeros(len(states))  # Initialize value-function

    # Doing three iterations of policy-evaluation.
    for _ in range(k):
        for s in states:
            # Evaluation of policy is easier than value-iteration, since you don't need to find the best action, you just pick the action determined by your policy.
            V[s] = sum(transition_probabilities[s][policy[s]][s_prime] * (reward_function[s][policy[s]][s_prime] + discount_factor * V[s_prime]) for s_prime in states)
    return V


def policy_improvement(V, states, actions, transition_probabilities, reward_function, discount_factor):
    
    policy = np.zeros(len(states), dtype=int)
    
    # For each state we have
    for s in states:
        
        action_values = np.zeros(len(actions))
        
        # We loop through the actions a and find argmax, basically we find the value of a which maximizes expected future reward for our state s.
        for a in actions:
            action_values[a] = sum(transition_probabilities[s][a][s_prime] * (reward_function[s][a][s_prime] + discount_factor * V[s_prime]) for s_prime in states)
        
        # We now assert that the new policy should pick the best action a (which we just calculated) when encountering state s.
        policy[s] = np.argmax(action_values)
    
    return policy


def policy_iteration(states, actions, transition_probabilities, reward_function, discount_factor, k, theta=0.0001):
    
    # Our initial policy is just to pick action 0 in every state.
    policy = np.zeros(len(states), dtype=int)  
    
    while True:

        # With this method call, we evaluate our policy k times.
        V = policy_evaluation_k(policy, states, transition_probabilities, reward_function, discount_factor, k)
        
        # At this point, we are updating our policy, taking argmax out of the actions A, for each state s.
        policy_prime = policy_improvement(V, states, actions, transition_probabilities, reward_function, discount_factor)
        
        max_diff = max( abs(policy[i] - policy_prime[i]) for i in range(len(policy)))
        policy = policy_prime.copy()
            
        if max_diff < theta:
            break

    
    # We are returning our policy, and our state-value-function which we connect to our policy.
    return policy, V
