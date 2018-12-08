
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


rows = cols = 5
states = rows * cols
actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]  # right, left, down, up


# In[3]:


# Maps indices of 2D array to 1D array index
def map_2d_to_1d(i, j):
    return i * cols + j

# Maps 1D array index to 2D array indices
def map_1d_to_2d(s):
    i = s // cols
    j = s % cols
    return i, j


# In[4]:


# Given current state and action taken, returns next state and reward 
def step(s, action):
    i, j = map_1d_to_2d(s)  # current state in the grid world

    # if current state is A, return A' and reward of +10
    if i == 0 and j == 1:
        return map_2d_to_1d(4, 1), 10.

    # if current state is B, return B' and reward of +5
    if i == 0 and j == 3:
        return map_2d_to_1d(2, 3), 5.

    i_ = i + action[0]
    j_ = j + action[1]

    # if not in (A or B) and agent doesn't jumps out of the grid, return next state(i_, j_), and reward value of 0
    if 0 <= i_ <= 4 and 0 <= j_ <= 4:
        return map_2d_to_1d(i_, j_), 0.
    # if not in (A or B) and agent jumps out of the grid, return current state(i, j), and reward value of -1
    return map_2d_to_1d(i, j), -1.


# In[5]:


k = 0.9  # discount rate
pi = 0.25  # probability of picking an action on any state

# We'll solve the linear system Ax=b
# where A is |states| x |states| 2D matrix containing coeff. of value function of each state
A = np.zeros((states, states))
b = np.zeros(states)

for s in range(states):
    # for each row of A, calculate the coefficient of v(s) for all s
    A[s, s] += 1
    for action in actions:
        s_, reward = step(s, action)
        b[s] += pi * reward
        A[s, s_] -= pi * k


# In[6]:


# Solve the linear system using numpy linalg library
value = np.linalg.solve(A, b)
value = np.round(value, 1)


# In[7]:


print("State value function:")
for i in range(rows):
    for j in range(cols):
        print(value[map_2d_to_1d(i, j)], end="       " if (value[map_2d_to_1d(i, j)] >= 0) else "     ")
    print("")

