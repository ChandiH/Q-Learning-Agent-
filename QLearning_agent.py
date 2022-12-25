import gym
import numpy as np
import random
from tqdm import tqdm
import Plot_Graph


class QAgent:

    def __init__(self, env, alpha, gamma):
        self.env = env

        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor

    def get_action(self, state):
        return np.argmax(self.q_table[state[0]])

    def update_parameters(self, state, action, reward, next_state):
        old_value = self.q_table[state[0], action]
        next_max = np.max(self.q_table[next_state])
        new_value = \
            old_value + \
            self.alpha * (reward + self.gamma * next_max - old_value)

        # update the q_table
        self.q_table[state[0], action] = new_value


def learning(n_episodes, epsilon, alpha, gamma):

    # For plotting metrics
    timesteps_per_episode = []
    penalties_per_episode = []

    for i in tqdm(range(0, n_episodes)):
        env = gym.make("Taxi-v3", render_mode='rgb_array').env
        agent = QAgent(env, alpha, gamma)
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        terminated, truncated = False, False

        while not terminated and not truncated:

            if random.uniform(0, 1) < epsilon:
                # Explore action space
                action = env.action_space.sample()
            else:
                # Exploit learned values
                action = agent.get_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)

            agent.update_parameters(state, action, reward, next_state)

            if reward == -10:
                penalties += 1

            state = (next_state, False)
            epochs += 1

        timesteps_per_episode.append(epochs)
        penalties_per_episode.append(penalties)
    print(timesteps_per_episode)
    print(f'Avg steps to complete ride: {np.array(timesteps_per_episode).mean()}')
    print(penalties_per_episode)
    print(f'Avg penalties to complete ride: {np.array(penalties_per_episode).mean()}')
    return timesteps_per_episode, penalties_per_episode


def loop(n_episodes,n_agents, epsilon, alpha, gamma):
    # For plotting metrics
    timesteps_per_agent = []
    penalties_per_agent = []

    for _ in range(n_agents):
        timesteps_per_episode, penalties_per_episode = learning(n_episodes, epsilon, alpha, gamma)
        timesteps_per_agent.append(timesteps_per_episode)
        penalties_per_agent.append(penalties_per_episode)

    return timesteps_per_agent, penalties_per_agent



"""
time_step, penalties = agent.learning(100, 0.1)
Plot_Graph.Draw.subplot_line_graph(time_step,"Time Step", penalties, "penalties", "QLearning")
print('action space {}'.format(agent.env.action_space))
print('State Space {}'.format(agent.env.observation_space))
"""
epsilon = 0.1
alpha = 1.0
gamma = 0.9
timesteps_agents, penalties_agents = learning(100, epsilon, alpha, gamma)

Plot_Graph.Draw.plot_line_graph(timesteps_agents, "Time Steps", "Q Agent - Time Steps")
Plot_Graph.Draw.plot_line_graph(penalties_agents, "Penalties", "Q Agent - Penalties")