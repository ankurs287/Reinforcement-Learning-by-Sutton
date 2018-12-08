
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[16]:


actions = [-1, +1]
TRUE_STATE_VALUES = np.array([0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0])


def step(state, action):
    reward = 0
    if state == 5 and action == +1:
        reward = 1
    return state + action, reward


def is_terminal(state):
    if state == 6 or state == 0:
        return True
    return False


# In[17]:


def td0(n_episodes=100, alpha=0.1):
    episodes = [1, 10, 100]
    state_values = np.zeros(7)
    state_values[1:6] = 0.5
    if alpha == 0.1:
        plt.figure()
        plt.plot(state_values[1:6], label=0)
    rmse = np.zeros(n_episodes + 1)
    for i in range(0, n_episodes + 1):
        state = 3
        while True:
            action = np.random.choice(actions)
            next_state, reward = step(state, action)
            state_values[state] += alpha * (reward + state_values[next_state] - state_values[state])
            state = next_state
            if is_terminal(state):
                break

        rmse[i] = np.sqrt(np.sum(np.power(TRUE_STATE_VALUES - state_values, 2)) / 5.0)

        if i in episodes and alpha == 0.1:
            plt.plot(state_values[1:6], label=i)
    if alpha == 0.1:
        plt.plot(TRUE_STATE_VALUES[1:6], label='TRUE VALUE')
        plt.xlabel('state')
        plt.ylabel('estimated value')
        plt.legend()
        plt.savefig('q6_1.png')
        plt.close()
    return rmse


# In[18]:


def alpha_mc(n_episodes=100, alpha=0.01):
    state_values = np.zeros(7)
    state_values[1:6] = 0.5
    rmse = np.zeros(n_episodes + 1)
    for i in range(n_episodes + 1):
        state = 3
        episode = []
        while True:
            action = np.random.choice(actions)
            next_state, reward = step(state, action)
            episode.append((state, action, reward))
            state = next_state
            if is_terminal(state):
                break
        g = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            g += reward
            state_values[state] += alpha * (g - state_values[state])
        rmse[i] = np.sqrt(np.sum(np.power(TRUE_STATE_VALUES - state_values, 2)) / 5.0)
    return rmse


# In[19]:


td_alphas = [0.1, 0.15, 0.05]
EPISODES = 100
plt.figure()
runs = 100
for alpha in td_alphas:
    td_rmse = np.zeros((runs, EPISODES + 1))
    for i in tqdm(range(runs)):
        td_rmse[i] = td0(n_episodes=EPISODES, alpha=alpha)

    plt.plot(np.mean(td_rmse, axis=0), label='TD with alpha=' + str(alpha))

mc_alphas = [0.01, 0.02, 0.04, 0.03]
for alpha in mc_alphas:
    mc_rmse = np.zeros((runs, EPISODES + 1))
    for i in tqdm(range(runs)):
        mc_rmse[i] = alpha_mc(n_episodes=EPISODES, alpha=alpha)

    plt.plot(np.mean(mc_rmse, axis=0), label='MC with alpha=' + str(alpha))

plt.xlabel('Walks/ Episodes')
plt.ylabel('Empirical RMSE, averaged over states')
plt.legend()
plt.savefig('q6_2.png')
plt.close()


# In[31]:




