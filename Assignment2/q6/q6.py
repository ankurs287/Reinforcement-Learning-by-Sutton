
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


rows = cols = 4
actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]  # right, left, down, up


# In[4]:


def display_value(value, off=1):
    print("State value function:")
    for i in range(rows):
        for j in range(cols):
            print(round(value[i, j], off), end="       " if (value[i, j] >= 0) else "     ")
        print("")


# In[5]:


def display_policy(pi):
    print("Policy:")
    for i in range(rows):
        for j in range(cols):
            if is_terminal_state(i, j):
                print("-----  |  ", end="")
                continue

            for k in range(len(actions)):
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
        print()


# In[6]:


def is_terminal_state(i, j):
    if i == j == 0 or i == j == rows-1:
        return True
    return False


# In[7]:


# Given current state and action taken, returns next state and reward 
def step(i, j, action):
    i_ = i + action[0]
    j_ = j + action[1]

    if 0 <= i_ < rows and 0 <= j_ < cols:
        return i_, j_, -1
    return i, j, -1.


# ###Policy Iteration

# In[12]:


def policy_iteration():
    value = np.zeros((rows, cols))  # state value function
    pi = np.zeros((rows, cols, len(actions)))
    pi.fill(0.25)

    while True:

        # Policy Evaluation
        theta = 1e-4  # parameter to check convergence
        itrtn = 0
        while True:
            delta = 0.
            old_value = value.copy()
            for i in range(rows):
                for j in range(cols):
                    if is_terminal_state(i, j):
                        continue
                    # for each state (i, j)
                    v = value[i, j]  # old value
                    v_ = 0.  # new value to be calculated using 
                    for a in range(len(actions)):
                        i_, j_, reward = step(i, j, actions[a])
                        v_ += 0.25 * (reward + value[i_, j_])
                    value[i, j] = v_
                    delta = max(delta, abs(v - v_))  # maximum change in new and old value of all states

            display_value(value, off=4)

            if itrtn % 10 >= 0:
                change_in_value = np.abs((old_value - value)).sum()
                print('------------------------------------')
                print('| Change in value: %f |' % change_in_value)
                print('------------------------------------')
            itrtn += 1

            if delta < theta:
                # value function converged to v of pi
                break

        # Policy Improvement
        optimal_policy = True
        for i in range(rows):
            for j in range(cols):
                old_policy = pi[i, j].copy()
                q_ = []
                for action in actions:
                    i_, j_, reward = step(i, j, action)
                    q_.append(round(reward + value[i_, j_], 2))

                pi[i, j] = np.zeros(len(actions))
                y = np.max(q_)
                z = q_.count(y)
                x = np.argsort(q_)
                for a in range(z):
                    pi[i, j, x[-a - 1]] = 1. / z

                if not np.count_nonzero(pi[i, j]) == np.count_nonzero(old_policy):
                    optimal_policy = False

        display_policy(pi)

        if optimal_policy:
            return value, pi


# In[13]:


value, pi = policy_iteration()


# In[14]:


display_value(value)
print()
display_policy(pi)


# ### Value Iteration

# In[15]:


def value_iteration():
    value = np.zeros((rows, cols))  # state value function
    theta = 1e-30  # parameter to check convergence
    while True:
        delta = 0.
        old_value = value.copy()
        for i in range(rows):
            for j in range(cols):
                if is_terminal_state(i, j):
                    continue
                # for each state (i, j)
                v = value[i, j]  # old value
                v_ = -float('inf')  # new value to be calculated using 
                for a in range(len(actions)):
                    i_, j_, reward = step(i, j, actions[a])
                    v_ = max(v_, reward + value[i_, j_])
                delta = max(delta, abs(v - v_))  # maximum change in new and old value of all states
                value[i, j] = v_

        change_in_value = np.abs((old_value - value)).sum()
        print('Change in value: %.2f' % change_in_value)
        display_value(value, off=4)

        if delta < theta:
            # value function converged to v of pi
            break

    pi = np.zeros((rows, cols, len(actions)))
    for i in range(rows):
        for j in range(cols):
            q_ = []
            for action in actions:
                i_, j_, reward = step(i, j, action)
                q_.append(round(reward + value[i_, j_], 2))

            y = np.max(q_)
            z = q_.count(y)
            x = np.argsort(q_)
            for a in range(z):
                pi[i, j, x[-a - 1]] = 1. / z

    display_policy(pi)

    return value, pi


# In[17]:


value, pi = value_iteration()
# `Change in value` is +ve in each iteration of VI


# `Change in value` is +ve in each iteration of VI

# In[18]:


display_value(value)
print()
display_policy(pi)

