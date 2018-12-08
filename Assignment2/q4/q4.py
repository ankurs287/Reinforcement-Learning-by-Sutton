
# coding: utf-8

# In[19]:


import numpy as np
from scipy import optimize as opt


# In[20]:


rows = cols = 5
states = rows * cols
actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]  # right, left, down, up
actions_size = len(actions)


# In[21]:


# Maps indices of 2D array to 1D array index
def map_2d_to_1d(i, j, cols=cols):
    return i * cols + j

# Maps 1D array index to 2D array indices
def map_1d_to_2d(s):
    i = s // cols
    j = s % cols
    return i, j


# In[22]:


# Given current state and action taken, returns next state and reward 
def step(s, action):
    i, j = map_1d_to_2d(s) # current state in the grid world

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


# In[23]:


k = 0.9  # discount rate

# Since we need to solve the |states| bellman optimality eqns which are non-linear. 
# We will convert our system of non-linear equations to system of linear inequalities
# s.t. we will have 4 * |states| linear inequalities and solve them, by finding the minimum
# value which satisfies all the inequalities, using scipy optimize library.

# Hence, we will solve the system: Ax >= b
# where A is (4 * |states|) x |states|2D matrix containing coeff. of value function of each state
A = np.zeros((4 * states, states))
b = np.zeros(4 * states)

for s in range(states):
    # for each row of A, calculate the coefficient of v(s) for all s
    for a in range(actions_size):
        s_, reward = step(s, actions[a])
        A[4 * s + a, s] -= 1
        A[4 * s + a, s_] += k
        b[4 * s + a] -= reward


# In[24]:


c = np.zeros(states)
c.fill(1)
x = opt.linprog(c=c, A_ub=A, b_ub=b)
value = np.round(x.x, 1)


# In[25]:


print("Optimal state-value function:")
for i in range(rows):
    for j in range(cols):
        print(value[map_2d_to_1d(i, j)], end="       " if (value[map_2d_to_1d(i, j)] >= 0) else "     ")
    print("")


# In[26]:


print("Optimal Policy:")
# Calculating Optimal Policy using optimal value function
pi = np.zeros((rows, cols, actions_size))
for s in range(states):
    q_ = []
    for action in actions:
        s_, reward = step(s, action)
        q_.append(value[s_])

    y = np.max(q_)
    z = q_.count(y)
    x = np.argsort(q_)
    i, j = map_1d_to_2d(s)
    for k in range(z):
        pi[i, j, x[-k - 1]] = 1. 
        
    if s > 0 and s % cols == 0:
        print()

    for k in range(actions_size):
        if pi[i, j, k] > 0:
            if k == 0:
                print("R, ", end="")
            elif k == 1:
                print("L, ", end="")
            elif k == 2:
                print("D, ", end="")
            elif k == 3:
                print("U, ", end="")
    print("  |  ", end="")

