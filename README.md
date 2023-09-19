# Reinforcement Learning Using OpenAI Gym

I used the OpenAI Gym's Taxi environment in this project. My main goal was to approach
the task using reinforcement learning techniques, ultimately improving the agent's
performance. I conducted in-depth analyses throughout the project, systematically
adjusting various parameters to assess and optimize the agent's efficiency


## Report and Analytics

I built two different agents. One is a random agent, which takes actions randomly, and the 
other one is a Q learning agent, which takes actions according to the q-value function.

![App Screenshot](/__reports__/Screenshot%202023-09-19%20065614.png)


The above graphs show how long it took him to complete his job, and the second graph 
shows how many penalties he received while performing his duties.
we can observe that, the random agent is NOT LEARNING anything, and he is taking actions
pure randomly.

![App Screenshot](/__reports__/Screenshot%202023-09-19%20065649.png)

This graph depicts three agents acting randomly and how the time and penalty for completing the task. We can clearly see that random agents are not learning.

![App Screenshot](/__reports__/Screenshot%202023-09-19%20065713.png)

However, after implementing the Q learning algorithm, we can clearly see that our agent learning and time required to complete the task are both decreasing, as are the penalties.


We can tune the agent by setting the learning rate (alpha) and discount rate (gamma) appropriately.
![App Screenshot](/__reports__/Screenshot%202023-09-19%20065738.png)
![App Screenshot](/__reports__/Screenshot%202023-09-19%20065809.png)

## Run Locally

Clone the project

```bash
  git clone https://github.com/ChandiH/Q-Learning-Agent-.git
```

Go to the project directory

```bash
  cd Q-Learning-Agent
```

Install libraries

```bash
  pip install numpy
  pip install gym
