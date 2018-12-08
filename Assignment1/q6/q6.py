
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# In[2]:


class MultiArmedBanditUCB:
    def __init__(self, k, epsilon=0., c=0.):
        self.k = k
        self.epsilon = epsilon
        self.c = c
        self.time = 0
        self.average_reward = 0

    def reset(self):
        self.time = 0
        self.average_reward = 0
        self.q_true = np.random.randn(self.k)
        self.q_estimate = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)

    def act(self):
        self.time += 1
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.k))
        if self.c != 0:
            UCB_estimation = self.q_estimate +                              self.c * np.sqrt(np.log(self.time) / (self.action_count + 0.00001))
            q_best = np.max(UCB_estimation)
            return np.random.choice([action for action, q in enumerate(UCB_estimation) if q == q_best])

        return np.argmax(self.q_estimate)

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.average_reward = (((self.time - 1.0) / self.time) * self.average_reward) + (reward / self.time)
        self.action_count[action] += 1
        self.q_estimate[action] += 1 / self.action_count[action] * (reward - self.q_estimate[action])
        return reward


# In[3]:


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
    return rewards.mean(axis=0), best_action_counts.mean(axis=0)


# In[4]:


runs = 2000
time = 1000
arms = 10

bandit0 = MultiArmedBanditUCB(arms, epsilon=0.1)
bandit1 = MultiArmedBanditUCB(arms, c=2)
bandit2 = MultiArmedBanditUCB(arms, c=1)
bandit3 = MultiArmedBanditUCB(arms, c=4)


# In[5]:


rewards0, _ = simulate(bandit0, runs, time)
rewards1, _ = simulate(bandit1, runs, time)
rewards2, _ = simulate(bandit2, runs, time)
rewards3, _ = simulate(bandit3, runs, time)


# In[6]:


plt.figure(figsize=(10, 30))
# average reward vs steps for c = 2
plt.subplot(3, 1, 1)
plt.xlabel('steps')
plt.ylabel('average reward')

plt.plot(rewards0, label='epsilon=0.1 greedy')
plt.plot(rewards1, label='UCB with c=2')
plt.legend()

# average reward vs steps for c = 1
plt.subplot(3, 1, 2)
plt.xlabel('steps')
plt.ylabel('average reward')

plt.plot(rewards0, label='epsilon=0.1 greedy')
plt.plot(rewards2, label='UCB with c=1')
plt.legend()

# average reward vs steps for c = 4
plt.subplot(3, 1, 3)
plt.xlabel('steps')
plt.ylabel('average reward')

plt.plot(rewards0, label='epsilon=0.1 greedy')
plt.plot(rewards3, label='UCB with c=4')
plt.legend()

plt.savefig('./figures/q6.png')

