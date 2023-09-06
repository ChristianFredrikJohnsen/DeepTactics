from QLearning_Acrobot import QLearningAgent
import wandb
import gymnasium as gym
import numpy as np

def discretize(state):
    step0 = 0.1
    step1 = 0.1
    step2 = 0.1
    step3 = 0.1
    step4 = 1.0
    step5 = 1.0

    return state[0]//step0, 0, state[2]//step2, 0, state[4]//step4, state[5]//step5

def train_agent(agent):
    train_env = gym.make(agent.cfg.env)
    eval_env = gym.make(agent.cfg.env, render_mode="human")

    wandb.init(project = agent.cfg.wandb_name, config = agent.cfg.get_members())

    for episode in range(1, agent.cfg.episodes + 1):

        obs, info = train_env.reset()
        episode_return = 0
        episode_length = 0

        while True:
            action = agent.act(discretize(obs))
            next_obs, reward, terminated, truncated, info = train_env.step(action)
            
            standardReward = reward
            ## reward = -1
            cos1 = next_obs[0]
            cos2 = next_obs[1]
            sin1 = next_obs[0]
            sin2 = next_obs[1]

            cos12 = cos1*cos2-sin1*sin2
            heightFromGround = (cos1 + cos12) / -2
            reward = heightFromGround

            agent.update_q_values(discretize(obs), reward, action, discretize(next_obs))

            obs = next_obs
            episode_return += standardReward
            episode_length += 1
            ## terminated betyr at spillet over, truncated betyr at spillets lengde har blitt nådd. 
            if terminated or truncated:
                break 
        
        wandb.log({"training return": episode_return})
        #print("Episode", episode, "episode return", episode_return, end = "")
        #print()

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
        #print()
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


        


