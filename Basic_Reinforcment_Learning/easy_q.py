from collections import defaultdict
import numpy as np
import gymnasium as gym

class Easy_Agent():

    def __init__(self, ac_dim, ob_dim, lr, epsilon, gamma, env, episodes, eval_frequency):
        
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = env
        self.episodes = episodes
        self.eval_frequency = eval_frequency

        self.q_values = defaultdict(lambda: [0] * self.ac_dim)
        self.num_updates = 0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return self.exploration_action(state)
        else:
            return self.greedy_action(state)

    def exploration_action(self, state):
        return np.random.randint(0, self.ac_dim)

    def greedy_action(self, state):
        return np.argmax(self.q_values[state])
    
    def update_q_values(self, state, reward, action, next_state, done):
        self.q_values[state][action] += self.lr * (reward + self.gamma * np.max(self.q_values[next_state])- self.q_values[state][action])
        #self.q_values[state][action] += self.lr * (reward + self.gamma * np.max(self.q_values[next_state]) * (1-done) - self.q_values[state][action])
        self.num_updates += 1

    def decay_lr(self):
        self.lr -= (0.5-0.01)/self.episodes

    def decay_epsilon(self):
        self.epsilon -= (0.25 - 0.05)/self.episodes
    
    def discretize(self, state):
        return tuple(np.round(state, 1))


    def train_agent(self):
        
        train_env = gym.make(self.env)
        eval_env = gym.make(self.env, render_mode = "human")

        for episode in range(1, self.episodes + 1):

            state, _ = train_env.reset()
            total_reward = 0

            while True:

                action = self.act(self.discretize(state))
                next_state, reward, terminated, truncated, _ = train_env.step(action)

                done = terminated or truncated
                self.update_q_values(self.discretize(state), reward, action, self.discretize(next_state), done)

                state = next_state
                total_reward += reward

                if terminated or truncated:
                    break
            
            self.decay_epsilon()
            self.decay_lr()
            print(self.q_values)
            print(self.num_updates)

            if episode % self.eval_frequency == 0:

                obs, info = eval_env.reset()
                episode_return = 0
                episode_length = 0
                


                ## Her må det være eval_env og ikke train_env, må se på dette.
                while True:
                    action = self.greedy_action(self.discretize(obs))
                    next_obs, reward, terminated, truncated, info = eval_env.step(action)

                    obs = next_obs
                    episode_return += reward
                    episode_length += 1
                    ## terminated betyr at spillet over, truncated betyr at spillets lengde har blitt nådd. 
                    if terminated or truncated:
                        break 
                print(episode_return)


james = Easy_Agent(ac_dim = 2, ob_dim = 4, lr = 0.5, epsilon = 0.25, gamma = 0.99, env = "CartPole-v1", episodes = 1, eval_frequency = 10)
james.train_agent()






