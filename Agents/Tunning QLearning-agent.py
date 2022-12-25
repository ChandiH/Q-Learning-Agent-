import gym
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from Graphs import Plot_Graph


class QAgent:

    def __init__(self, alpha, gamma):
        self.env = gym.make("Taxi-v3", render_mode='rgb_array').env

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

    def learning(self, n_episodes, epsilon):
        # For plotting metrics
        timesteps_per_episode = []
        penalties_per_episode = []

        for i in tqdm(range(0, n_episodes)):

            state = self.env.reset()

            epochs, penalties, reward, = 0, 0, 0
            terminated, truncated = False, False

            while not terminated and not truncated:

                if random.uniform(0, 1) < epsilon:
                    # Explore action space
                    action = self.env.action_space.sample()
                else:
                    # Exploit learned values
                    action = agent.get_action(state)

                next_state, reward, terminated, truncated, info = self.env.step(action)

                agent.update_parameters(state, action, reward, next_state)

                if reward == -10:
                    penalties += 1

                state = (next_state, False)
                epochs += 1

            timesteps_per_episode.append(epochs)
            penalties_per_episode.append(penalties)
        return timesteps_per_episode, penalties_per_episode

    def plot_graph(self, timeStep, Penalties):
        print(timeStep)
        print(Penalties)

        # Create a figure with a larger size
        plt.figure(figsize=(10, 6))

        # Create a subplot for the first plot
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(timeStep)
        ax1.set_title("timesteps_per_episode")

        # Create a subplot for the second plot
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(Penalties)
        ax2.set_title("penalties_per_episode")

        # Save the plot as a high-resolution image file
        plt.savefig("QAgent.png", dpi=300)

        # Show the figure
        plt.show()

        print(f'Avg steps to complete ride: {np.array(timeStep).mean()}')
        print(f'Avg penalties to complete ride: {np.array(Penalties).mean()}')


alphas = [0.75]
gamma = [0.9, 0.75, 0.5, 0.25, 0.1]
timeSteps_set = []
penalties_set = []

for a in alphas:
    for g in gamma:
        agent = QAgent(a, g)
        time_step, penalties = agent.learning(500, 0.1)
        timeSteps_set.append(time_step)
        penalties_set.append(penalties)

plt.figure(figsize=(10, 6))

Plot_Graph.Draw.plot_multiple_line_graph(timeSteps_set, f"Time Step - Gamma = {gamma}", "Tunning Q Agent - Time Step")
Plot_Graph.Draw.plot_multiple_line_graph(penalties_set, f"Penalties - Gamma = {gamma}", "Tunning Q Agent - Penalties")
