
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# In[5]:


class MultiArmedBanditGradientAlgorithm:
    def __init__(self, k, alpha, baseline, mu):
        self.k = k
        self.alpha = alpha
        self.baseline = baseline
        self.mu = mu

    def reset(self):
        self.time = 0
        self.average_reward = 0
        self.q_true = np.random.randn(self.k) + self.mu
        self.q_estimate = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)

    def act(self):
        pref = np.exp(self.q_estimate)
        self.action_prob = pref / np.sum(pref)
        action = np.random.choice(self.k, p=self.action_prob)
        self.action_count[action] += 1
        return action

    def step(self, action):
        self.time += 1
        reward = np.random.randn() + self.q_true[action]
        self.average_reward = (self.time - 1.0) / self.time * self.average_reward + reward / self.time
        rt = 0
        if self.baseline:
            rt = self.average_reward

        one_hot = np.zeros(self.k)
        one_hot[action] = 1
        self.q_estimate += self.alpha * (reward - rt) * (one_hot - self.action_prob)

        return reward


# In[6]:


def simulate(bandit, runs, time):
    best_action_counts = np.zeros((runs, time))
    for r in tqdm(range(runs)):
        bandit.reset()
        for t in range(time):
            action = bandit.act()
            bandit.step(action)
            if action == bandit.best_action:
                best_action_counts[r, t] = 1
    return best_action_counts.mean(axis=0)


# In[7]:


bandit0 = MultiArmedBanditGradientAlgorithm(10, 0.1, True, 4)
bandit1 = MultiArmedBanditGradientAlgorithm(10, 0.1, False, 4)
bandit2 = MultiArmedBanditGradientAlgorithm(10, 0.4, True, 4)
bandit3 = MultiArmedBanditGradientAlgorithm(10, 0.4, False, 4)


# In[8]:


runs = 2000
time = 1000
best_action_counts0 = simulate(bandit0, runs, time)
best_action_counts1 = simulate(bandit1, runs, time)
best_action_counts2 = simulate(bandit2, runs, time)
best_action_counts3 = simulate(bandit3, runs, time)


# In[9]:


plt.xlabel('Steps')
plt.ylabel('% Optimal action')

plt.plot(best_action_counts0, label='alpha = 0.1 w/ baseline')
plt.plot(best_action_counts1, label='alpha = 0.1 w/o baseline')
plt.plot(best_action_counts2, label='alpha = 0.4 w/ baseline')
plt.plot(best_action_counts3, label='alpha = 0.4 w/0 baseline')

plt.legend()

plt.savefig('./figures/q7.png')
plt.close()

