from DQNAgent import DQNAgent
import gymnasium as gym


def train_agent(agent):

    train_env = gym.make(agent.cfg.env)
    eval_env = gym.make(agent.cfg.env, render_mode="human")

    for episode in range(1, agent.cfg.episodes + 1):

        obs, _ = train_env.reset()
        episode_return, episode_length = 0, 0; losses = []

        while True:

            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)

            episode_return += reward; episode_length += 1

            agent.store_transition(obs, action, reward, next_obs, truncated or terminated)            
            loss = agent.update_q_values()
            losses.append(loss.item()) if loss is not None else None

            obs = next_obs

            if terminated or truncated:
                break
        
        if episode % 10 == 0:
            print("Episode:", episode, "episode return:", episode_return, end="\t") if not losses else print("Episode:", episode, "episode return:", episode_return, "loss:", loss.item(), '\n')

        if episode % agent.cfg.update_target_network_freq == 0:
            agent.update_target_network()

        
        if episode % agent.cfg.eval_freq == 0:

            obs, _ = eval_env.reset()
            episode_return, episode_length = 0, 0

            while True:
                action = agent.greedy_action(obs)
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)

                obs = next_obs

                episode_return += reward; episode_length += 1

                if terminated or truncated:
                    break

            print("Eval return:", episode_return, end="")

if __name__ == '__main__':

    agent = DQNAgent(DQNAgent.Config())
    train_agent(agent)

    agent.save("Advanced_Reinforcement_Learning/DQN_no_logging/testdqn.pyt")