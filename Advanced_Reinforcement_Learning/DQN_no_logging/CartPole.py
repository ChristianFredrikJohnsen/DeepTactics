import wandb
from DQNAgent import DQNAgent
import torch
import gymnasium as gym
import numpy as np


def train_agent(agent):

    train_env = gym.make(agent.cfg.env)
    eval_env = gym.make(agent.cfg.env, render_mode="human")

    # Logging
    wandb.init(project = agent.cfg.wandb_name, config = agent.cfg.get_members())

    for episode in range(1, agent.cfg.episodes + 1):

        obs, _ = train_env.reset()

        episode_return = 0
        episode_length = 0
        losses = []

        while True:

            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)

            episode_return += reward
            episode_length += 1

            # Save transition
            agent.store_transition(obs, action, reward, next_obs, truncated or terminated)
            
            # Update DQN
            loss = agent.update_q_values()

            # If we have not yet sampled enough transitions, then there will be no loss.
            # This is because we are at a stage where we are just filling up our buffer with state-transitions.
            if loss is not None:
                
                # If the number of transitions in the buffer is at or above minimum, then we actually do something.
                losses.append(loss.item())

            obs = next_obs

            if terminated or truncated:
                break
            
        # This block will only be executed if the number of state-action transitions in the buffer size is less than minimum.
        # In this case, there is no loss to log, since we are not calculating gradient yet.
        # We are just sampling transitions to fill up the buffer at this point, so all we do is to report the episode return and length.
        if not losses:
            wandb.log({"training return": episode_return, "train episode length": episode_length})
        
        # In this case, the number of transitions in the buffer is at or above minimum. For each action we take, there is some loss with respect 
        # to the loss function. The loss is calculated by sampling some transitions from the buffer.
        else:
            losses = np.average(losses)
            wandb.log({"training return": episode_return, "train episode length": episode_length, "loss": losses})

        print("Episode", episode, "episode return:", episode_return, end="\t")

        # Update the target network if the training episode which wsa just finished, turns out to
        # be a multiple of the update frequency for target network.
        if episode % agent.cfg.update_target_network_freq == 0:
            agent.update_target_network()

        # This is just going to give us an evaluation of the current policy
        # We always act greedily, and this procedure is only done every x episodes,
        # where x is the eval frequency.
        if episode % agent.cfg.eval_freq == 0:

            obs, _ = eval_env.reset()
            episode_return = 0
            episode_length = 0

            while True:
                
                action = agent._greedy_action(obs)
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)

                obs = next_obs

                episode_return += reward
                episode_length += 1

                if terminated or truncated:
                    break

            wandb.log({"eval return": episode_return, "eval episode length": episode_length})
            print("Eval return:", episode_return, end="")
        
        print()

    wandb.finish()


if __name__ == '__main__':

    agent = DQNAgent(DQNAgent.Config())
    train_agent(agent)

    agent.save("testdqn.pyt")