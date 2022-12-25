import gym
from tqdm import tqdm
import numpy as np
import Plot_Graph


class RandomAgent:
    """ selects actions randomly. """

    def __init__(self, env):
        self.env = env

    def get_action(self, state) -> int:
        return self.env.action_space.sample()


def run(n_episodes):
    env = gym.make("Taxi-v3", render_mode='rgb_array').env
    agent = RandomAgent(env)

    timesteps_per_episode = []
    penalties_per_episode = []

    for i in tqdm(range(0, n_episodes)):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        terminated, truncated = False, False

        while not terminated and not truncated:

            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        timesteps_per_episode.append(epochs)
        penalties_per_episode.append(penalties)
    print(timesteps_per_episode)
    print(f'Avg steps to complete ride: {np.array(timesteps_per_episode).mean()}')
    print(penalties_per_episode)
    print(f'Avg penalties to complete ride: {np.array(penalties_per_episode).mean()}')
    return timesteps_per_episode, penalties_per_episode


def loop(n_episodes, n_agents):
    timesteps_per_agent = []
    penalties_per_agent = []

    for _ in range(n_agents):
        time, penalty = run(n_episodes)
        timesteps_per_agent.append(time)
        penalties_per_agent.append(penalty)

    return timesteps_per_agent, penalties_per_agent


# For plotting metrics
timesteps_agents, penalties_agents = loop(100, 1)

Plot_Graph.Draw.subplot_line_graph(timesteps_agents, "Time Steps", penalties_agents, "Penalties", "random_agent")

#Plot_Graph.Draw.plot_multiple_line_graph(timesteps_agents, "Time Steps", "Random Agent - Time Steps")
#Plot_Graph.Draw.plot_multiple_line_graph(penalties_agents, "Penalties", "Random Agent - Penalties")
