
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# In[14]:


class MultiArmedBanditNonStationaryEpsilonGreedy:
    def __init__(self, k, epsilon, alpha):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha

    def reset(self):
        self.time = 0
        self.q_true = np.random.randn(self.k)
        self.q_estimate = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.k))
        return np.argmax(self.q_estimate)

    def step(self, action):
        self.time += 1
        reward = np.random.randn() + self.q_true[action]
        self.action_count[action] += 1
        self.q_estimate[action] += self.alpha * (reward - self.q_estimate[action])
        return reward


# In[15]:


def simulate(bandit, runs, time):
    best_action_counts = np.zeros((runs, time))
    rewards = np.zeros((runs, time))
    for r in tqdm(range(runs)):
        bandit.reset()
        for t in range(time):
            action = bandit.act()
            if action == bandit.best_action:
                best_action_counts[r, t] = 1
            rewards[r, t] = bandit.step(action)
            for i in range(bandit.k):
                if bandit.q_true[i] < 0.5:
                    bandit.q_true[i] += 0.01
                else:
                    bandit.q_true[i] -= 0.01
    return rewards.mean(axis=0), best_action_counts.mean(axis=0)


# In[16]:


runs = 2000
time = 10000
arms = 10

bandit = MultiArmedBanditNonStationaryEpsilonGreedy(arms, 0.1, 0.1)


# In[17]:


rewards, best_action_counts = simulate(bandit, runs, time)


# In[18]:


plt.figure(figsize=(10, 20))

# average reward vs steps
plt.subplot(2, 1, 1)
plt.xlabel('steps')
plt.ylabel('average reward')

plt.plot(rewards, label='epsilon = 0.1 and alpha = 0.1')
plt.legend()

# optimal action vs steps
plt.subplot(2, 1, 2)
plt.xlabel('steps')
plt.ylabel('% optimal action')

plt.plot(best_action_counts, label='epsilon = 0.1 and alpha = 0.1')
plt.legend()

plt.savefig('./figures/q5.png')
plt.close()

