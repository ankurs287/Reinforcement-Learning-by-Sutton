
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# In[2]:


class MultiArmedBanditStationaryEpsilonGreedy:
    def __init__(self, k, epsilon=1., var=1, constant=True):
        self.k = k
        self.epsilon = epsilon
        self.constant = constant
        self.time = 0
        self.average_reward = 0
        self.var = var

    def reset(self):
        self.time = 0
        self.average_reward = 0
        self.q_true = self.var * np.random.randn(self.k)
        self.q_estimate = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        if not self.constant:
            self.epsilon = 1.

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.k))
        return np.argmax(self.q_estimate)

    def step(self, action):
        self.time += 1
        if not self.constant:
            self.epsilon /= (self.time + 1)
        reward = np.random.randn() + self.q_true[action]
        self.average_reward = (((self.time - 1.0) / self.time) * self.average_reward) + (reward / self.time)
        self.action_count[action] += 1
        self.q_estimate[action] += 1 / self.action_count[action] * (reward - self.q_estimate[action])
        return reward


# In[3]:


def simulate(bandit, runs, time):
    abs_estimation_error = np.zeros((bandit.k, runs, time))
    best_action_counts = np.zeros((runs, time))
    rewards = np.zeros((runs, time))
    for r in tqdm(range(runs)):
        bandit.reset()
        for t in range(time):
            action = bandit.act()
            if action == bandit.best_action:
                best_action_counts[r, t] = 1
            rewards[r, t] = bandit.step(action)
            for arm in range(bandit.k):
                abs_estimation_error[arm, r, t] = abs(bandit.q_estimate[arm] - bandit.q_true[arm])
    return rewards.mean(axis=0), best_action_counts.mean(axis=0), abs_estimation_error.mean(axis=1)


# In[4]:


# var = 1

runs = 2000
time = 1000
arms = 10

bandit0 = MultiArmedBanditStationaryEpsilonGreedy(arms, epsilon=0)
bandit1 = MultiArmedBanditStationaryEpsilonGreedy(arms, epsilon=0.01)
bandit2 = MultiArmedBanditStationaryEpsilonGreedy(arms, epsilon=0.1)
bandit3 = MultiArmedBanditStationaryEpsilonGreedy(arms, constant=False)


# In[5]:


rewards0, best_action_counts0, abs_estimation_error0 = simulate(bandit0, runs, time)
rewards1, best_action_counts1, abs_estimation_error1 = simulate(bandit1, runs, time)
rewards2, best_action_counts2, abs_estimation_error2 = simulate(bandit2, runs, time)
rewards3, best_action_counts3, abs_estimation_error3 = simulate(bandit3, runs, time)


# In[6]:


plt.figure(figsize=(10, 60))

# average reward vs steps
plt.subplot(6, 1, 1)
plt.xlabel('steps')
plt.ylabel('average reward')

plt.plot(rewards0, label='epsilon = 0')
plt.plot(rewards1, label='epsilon = 0.01')
plt.plot(rewards2, label='epsilon = 0.1')
plt.plot(rewards3, label='epsilon(t) = 1/t')
plt.legend()

# optimal action vs steps
plt.subplot(6, 1, 2)
plt.xlabel('steps')
plt.ylabel('% optimal action')

plt.plot(best_action_counts0, label='epsilon = 0')
plt.plot(best_action_counts1, label='epsilon = 0.01')
plt.plot(best_action_counts2, label='epsilon = 0.1')
plt.plot(best_action_counts3, label='epsilon(t) = 1/t')
plt.legend()

# average absolute error in the estimate vs steps for epsilon = 0
plt.subplot(6, 1, 3)
plt.xlabel('steps')
plt.ylabel('average absolute error in the estimate for epsilon = 0')
for arm in range(arms):
    plt.plot(abs_estimation_error0[arm], label='arm = %s' % arm)
plt.legend()

# average absolute error in the estimate vs steps for epsilon = 0.01
plt.subplot(6, 1, 4)
plt.xlabel('steps')
plt.ylabel('average absolute error in the estimate for epsilon = 0.01')
for arm in range(arms):
    plt.plot(abs_estimation_error1[arm], label='arm = %s' % arm)
plt.legend()

# average absolute error in the estimate vs steps for epsilon = 0.1
plt.subplot(6, 1, 5)
plt.xlabel('steps')
plt.ylabel('average absolute error in the estimate for epsilon = 0.1')
for arm in range(arms):
    plt.plot(abs_estimation_error2[arm], label='arm = %s' % arm)
plt.legend()

# average absolute error in the estimate vs steps for epsilon(t) = 1/t
plt.subplot(6, 1, 6)
plt.xlabel('steps')
plt.ylabel('average absolute error in the estimate for epsilon(t) = 1/t')
for arm in range(arms):
    plt.plot(abs_estimation_error3[arm], label='arm = %s' % arm)
plt.legend()

plt.savefig('./figures/q1.png')
plt.close()


# In[10]:


var = 2

runs = 2000
time = 1000
arms = 10

bandit0 = MultiArmedBanditStationaryEpsilonGreedy(arms, epsilon=0, var=var)
bandit1 = MultiArmedBanditStationaryEpsilonGreedy(arms, epsilon=0.01, var=var)
bandit2 = MultiArmedBanditStationaryEpsilonGreedy(arms, epsilon=0.1, var=var)
bandit3 = MultiArmedBanditStationaryEpsilonGreedy(arms, var=var, constant=False)


# In[11]:


rewards0, best_action_counts0, abs_estimation_error0 = simulate(bandit0, runs, time)
rewards1, best_action_counts1, abs_estimation_error1 = simulate(bandit1, runs, time)
rewards2, best_action_counts2, abs_estimation_error2 = simulate(bandit2, runs, time)
rewards3, best_action_counts3, abs_estimation_error3 = simulate(bandit3, runs, time)


# In[12]:


plt.figure(figsize=(10, 60))

# average reward vs steps
plt.subplot(6, 1, 1)
plt.xlabel('steps')
plt.ylabel('average reward')

plt.plot(rewards0, label='epsilon = 0')
plt.plot(rewards1, label='epsilon = 0.01')
plt.plot(rewards2, label='epsilon = 0.1')
plt.plot(rewards3, label='epsilon(t) = 1/t')
plt.legend()

# optimal action vs steps
plt.subplot(6, 1, 2)
plt.xlabel('steps')
plt.ylabel('% optimal action')

plt.plot(best_action_counts0, label='epsilon = 0')
plt.plot(best_action_counts1, label='epsilon = 0.01')
plt.plot(best_action_counts2, label='epsilon = 0.1')
plt.plot(best_action_counts3, label='epsilon(t) = 1/t')
plt.legend()

# average absolute error in the estimate vs steps for epsilon = 0
plt.subplot(6, 1, 3)
plt.xlabel('steps')
plt.ylabel('average absolute error in the estimate for epsilon = 0')
for arm in range(arms):
    plt.plot(abs_estimation_error0[arm], label='arm = %s' % arm)
plt.legend()

# average absolute error in the estimate vs steps for epsilon = 0.01
plt.subplot(6, 1, 4)
plt.xlabel('steps')
plt.ylabel('average absolute error in the estimate for epsilon = 0.01')
for arm in range(arms):
    plt.plot(abs_estimation_error1[arm], label='arm = %s' % arm)
plt.legend()

# average absolute error in the estimate vs steps for epsilon = 0.1
plt.subplot(6, 1, 5)
plt.xlabel('steps')
plt.ylabel('average absolute error in the estimate for epsilon = 0.1')
for arm in range(arms):
    plt.plot(abs_estimation_error2[arm], label='arm = %s' % arm)
plt.legend()

# average absolute error in the estimate vs steps for epsilon(t) = 1/t
plt.subplot(6, 1, 6)
plt.xlabel('steps')
plt.ylabel('average absolute error in the estimate for epsilon(t) = 1/t')
for arm in range(arms):
    plt.plot(abs_estimation_error3[arm], label='arm = %s' % arm)
plt.legend()

plt.savefig('./figures/q2.png')
plt.close()

