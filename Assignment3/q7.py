
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# In[4]:


COLS = 12
ROWS = 4
START_STATE = 36
EPSILON = 0.1
ACTIONS = [[0, 1], [0, -1], [1, 0], [-1, 0]]  # right, left, down, up
ALPHA = 0.5


# In[5]:


def is_terminal(s):
    if s == (COLS * ROWS) - 1:
        return True
    return False


def is_cliff(i, j):
    if i == 3 and 1 <= j <= COLS - 2:
        return True
    return False


# In[6]:


# Maps indices of 2D array to 1D array index
def map_2d_to_1d(i, j):
    return i * COLS + j


# Maps 1D array index to 2D array indices
def map_1d_to_2d(s):
    i = s // COLS
    j = s % COLS
    return i, j


# In[7]:


def policy(s, q):
    if np.random.uniform() <= EPSILON:
        a = np.random.randint(4)
        return a
    else:
        return np.argmax(q[s])


# In[8]:


def step(s, a):
    action = ACTIONS[a]
    i, j = map_1d_to_2d(s)
    i_ = i + action[0]
    j_ = j + action[1]
    if i_ < 0 or j_ < 0 or j_ >= COLS or i_ >= ROWS:
        return s, -1
    if is_cliff(i_, j_):
        return START_STATE, -100
    return map_2d_to_1d(i_, j_), -1


# In[9]:


def sarsa(n_episodes):
    q = np.zeros((ROWS * COLS, 4))
    rewards = np.zeros(n_episodes)
    for i in range(n_episodes):
        s = START_STATE
        a = policy(s, q)
        while True:
            s_, r = step(s, a)
            rewards[i] += r
            a_ = policy(s_, q)
            q[s, a] += ALPHA * (r + q[s_, a_] - q[s, a])
            s = s_
            a = a_
            if is_terminal(s):
                break
        rewards[i] = max(rewards[i], -100)
    return rewards


# In[10]:


def q_learning(n_episodes):
    rewards = np.zeros(n_episodes)
    q = np.zeros((ROWS * COLS, 4))
    for i in range(n_episodes):
        s = START_STATE
        while True:
            a = policy(s, q)
            s_, r = step(s, a)
            rewards[i] += r
            q[s, a] += ALPHA * (r + np.max(q[s_]) - q[s, a])
            s = s_
            if is_terminal(s):
                break
        rewards[i] = max(rewards[i], -100)
    return rewards


# In[11]:


EPISODES = 500
runs = 200

sum_sarsa = np.zeros((runs, EPISODES))
sum_q_learning = np.zeros((runs, EPISODES))
for r in tqdm(range(runs)):
    sum_sarsa[r] = sarsa(EPISODES)
    sum_q_learning[r] = q_learning(EPISODES)

plt.plot(np.mean(sum_sarsa, axis=0), label='Sarsa')
plt.plot(np.mean(sum_q_learning, axis=0), label='Q-Learning')
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')
plt.ylim([-100, 0])
plt.legend()
plt.savefig('q7.png')
plt.close()

