from QLearning_CartPole_lambda import QLearningAgent
import wandb
import gymnasium as gym
import numpy as np
from collections import defaultdict


def discretize(state):
    return tuple(np.round(state, 1))

def train_agent(agent):
    train_env = gym.make(agent.cfg.env)
    eval_env = gym.make(agent.cfg.env, render_mode="human")

    # wandb.init(project = agent.cfg.wandb_name, config = agent.cfg.get_members())

    for episode in range(1, agent.cfg.episodes + 1):

        state, info = train_env.reset()
        episode_return = 0
        episode_length = 0
        E = {}
        state = discretize(state)

        while True:

            
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            next_state = discretize(next_state)


            if (action == np.argmax(agent.q_values[state])):
                E = defaultdict(int, {key: eligibility * agent.lamb * agent.gamma for key, eligibility in E.items()})
            else:
                E = defaultdict(int)

            E[(state, action)] += 1

            done = terminated or truncated
            agent.update_q_values(state, reward, action, next_state, E, done)

            state = next_state
            episode_return += reward
            episode_length += 1
            
            ## terminated betyr at spillet over, truncated betyr at spillets lengde har blitt nådd. 
            if terminated or truncated:
                break 
        
        wandb.log({"training return": episode_return})

        if episode % agent.cfg.eval_frequency == 0:

            state, info = eval_env.reset()
            episode_return = 0
            episode_length = 0
            
            ## Her må det være eval_env og ikke train_env, må se på dette.
            while True:
                
                action = agent._greedy_action(discretize(state))
                next_state, reward, terminated, truncated, info = eval_env.step(action)

                state = next_state
                episode_return += reward
                episode_length += 1
                
                ## terminated betyr at spillet over, truncated betyr at spillets lengde har blitt nådd. 
                if terminated or truncated:
                    break 
            
            wandb.log({"eval return": episode_return})
            print(f'Episode: {episode}, Episode return: {episode_return}', end = "")
            print()
    wandb.finish()


if __name__ == '__main__':

    agent = QLearningAgent(QLearningAgent.Config())
    train_agent(agent)
    agent.save(str(agent.cfg.env) + ".agent")

    # env = gym.make("CartPole-v1", render_mode = "human")
    # observation, info = env.reset()

    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     next_state, reward, terminate, truncated, inf = env.step(action)


        


















