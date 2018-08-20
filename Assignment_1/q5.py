# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from multi_armed_bandit import MultiArmBandit


# In[7]:


def simulate(bandit, runs, time):
    bandit.best_action_counts = np.zeros((runs, time))
    bandit.rewards = np.zeros((runs, time))
    for r in tqdm(range(runs)):
        bandit.reset()
        for t in range(time):
            bandit.time += 1
            action = bandit.act()
            if action == bandit.best_arm:
                bandit.best_action_counts[r, t] = 1
            reward = bandit.step(action)
            bandit.rewards[r, t] = reward
            for i in range(len(bandit.q_true)):
                bandit.q_true[i] += 0.01 if bandit.q_true[i] < 0.5 else -0.01
    bandit.rewards = bandit.rewards.mean(axis=0)  # taking average of all the runs
    bandit.best_action_counts = bandit.best_action_counts.mean(axis=0)  # taking average of all the runs
    return bandit.rewards, bandit.best_action_counts


# In[8]:


runs = 2000
time = 10000
arms = 10

bandit0 = MultiArmBandit(arms, epsilon=0.1)

# In[9]:


rewards0, best_action_counts0 = simulate(bandit0, runs, time)

# In[1]:


plt.figure(figsize=(20, 30))

# average reward vs steps
plt.subplot(3, 2, 1)
plt.xlabel('steps')
plt.ylabel('average reward')

plt.plot(rewards0, label='epsilon = 0.1')
plt.legend()

# optimal action vs steps
plt.subplot(3, 2, 2)
plt.xlabel('steps')
plt.ylabel('% optimal action')

plt.plot(best_action_counts0, label='epsilon = 0.1')
plt.legend()

plt.savefig('./q5.png')
plt.close()
