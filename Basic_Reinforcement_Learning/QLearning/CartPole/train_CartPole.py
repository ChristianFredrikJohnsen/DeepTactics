from QLearning_CartPole import QLearningAgent
import wandb
import gymnasium as gym
import numpy as np


def discretize(state):
    return tuple(np.round(state, 1))

def train_agent(agent):
    train_env = gym.make(agent.cfg.env)
    eval_env = gym.make(agent.cfg.env, render_mode="human")

    wandb.init(project = agent.cfg.wandb_name, config = agent.cfg.dict)

    for episode in range(1, agent.cfg.episodes + 1):

        obs, info = train_env.reset()
        episode_return = 0
        episode_length = 0

        while True:

            action = agent.act(discretize(obs))
            next_obs, reward, terminated, truncated, info = train_env.step(action)

            
            agent.update_q_values(discretize(obs), reward, action, discretize(next_obs))

            obs = next_obs
            episode_return += reward
            episode_length += 1


            ## terminated betyr at spillet over, truncated betyr at spillets lengde har blitt nådd. 
            if terminated or truncated:
                break 
        
        wandb.log({"training return": episode_return})
        # print(f'Episode: {episode}, Episode return: {episode_return}', end = "")
        # print()

        if episode % agent.cfg.eval_frequency == 0:

            obs, info = eval_env.reset()
            episode_return = 0
            episode_length = 0
            


            ## Her må det være eval_env og ikke train_env, må se på dette.
            while True:
                action = agent._greedy_action(discretize(obs))
                next_obs, reward, terminated, truncated, info = eval_env.step(action)

                obs = next_obs
                episode_return += reward
                episode_length += 1
                ## terminated betyr at spillet over, truncated betyr at spillets lengde har blitt nådd. 
                if terminated or truncated:
                    break 
            
            wandb.log({"eval return": episode_return})
            print("Episode", episode, "episode return", episode_return, end = "")
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


        


















