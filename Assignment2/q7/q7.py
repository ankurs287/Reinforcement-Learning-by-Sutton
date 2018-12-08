# coding: utf-8

# In[20]:


from math import exp, factorial

import numpy as np

# In[21]:
print("If this code doesn't work please try running notebook")


MAX_CARS = 20
states = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
actions = np.arange(-MAX_CARS, MAX_CARS + 1)

# In[22]:


RENT_REWARD = 10
MOVE_COST = 2
MAX_MOVES = 5
EXPECTED_REQUESTS = [3, 4]
EXPECTED_RETURNS = [3, 2]
k = 0.9
MAX_REQUESTS = 8
MAX_RETURNS = 8

# In[23]:


poisson_pmf = np.zeros((15, 15))
poisson_pmf.fill(-1)


# In[24]:


def poisson(l, n):
    global poisson_pmf
    if poisson_pmf[l, n] != -1:
        return poisson_pmf[l, n]
    poisson_pmf[l, n] = exp(-l) * pow(l, n) / factorial(n)
    return poisson_pmf[l, n]


# In[25]:


def step(i, j, action, value):
    cars_moved = action
    expected_return = -abs(cars_moved if cars_moved <= 0 else cars_moved - 1) * MOVE_COST

    for d1 in range(MAX_REQUESTS + 1):
        for d2 in range(MAX_REQUESTS + 1):
            for e1 in range(MAX_RETURNS + 1):
                for e2 in range(MAX_RETURNS + 1):
                    p = poisson(EXPECTED_REQUESTS[0], d1) * poisson(EXPECTED_REQUESTS[1], d2) * poisson(
                        EXPECTED_RETURNS[0], e1) * poisson(EXPECTED_RETURNS[1], e2)

                    # Cars after moving
                    i_ = int(min(i - cars_moved, MAX_CARS))
                    j_ = int(min(j + cars_moved, MAX_CARS))

                    d1_ = int(min(d1, i_))
                    d2_ = int(min(d2, j_))
                    # Remaining cars after renting them
                    i_ -= d1_
                    j_ -= d2_

                    # Cars after returns
                    i_ = int(min(i_ + e1, MAX_CARS))
                    j_ = int(min(j_ + e2, MAX_CARS))

                    reward = (d1_ + d2_) * RENT_REWARD
                    if i_ > 10:
                        reward -= 4
                    if j_ > 10:
                        reward -= 4

                    expected_return += p * (reward + k * value[i_, j_])

    return expected_return


# In[26]:


def policy_iteration():
    iterations = 0
    value = np.zeros(states.shape)
    pi = np.zeros(states.shape, dtype=np.int)  # store the action to be taken at a particular state

    while True:

        print("iterations: ", iterations)

        # Policy Evaluation
        theta = 1e-5  # parameter to check convergence
        count = 0
        print("count: ")
        while True:
            delta = 0.
            count += 1
            print(count)
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    v = value[i, j]  # old value
                    value[i, j] = step(i, j, pi[i, j], value)
                    delta = max(delta, abs(v - value[i, j]))

            if delta < theta:
                print()
                break

        # Policy Improvement
        optimal_policy = True
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_policy = pi[i, j].copy()
                q_ = []
                for action in actions:
                    if (0 <= action <= i) or (action < 0 and j >= abs(action)):
                        q_.append(step(i, j, action, value))
                    else:
                        q_.append(-float('inf'))
                pi[i, j] = actions[np.argmax(q_)]

                if not pi[i, j] == old_policy:
                    optimal_policy = False

        if optimal_policy:
            return value, pi
        iterations += 1


# In[27]:


value, pi = policy_iteration()

# In[ ]:


print('Optimal Policy:')
for i in range(MAX_CARS + 1):
    for j in range(MAX_CARS + 1):
        print(pi[i, j], end="    " if pi[i, j] > 0 else "   ")
    print()

print()

print('Optimal Value Function:')
for i in range(MAX_CARS + 1):
    for j in range(MAX_CARS + 1):
        print(round(value[i, j], 1), end="    " if value[i, j] > 0 else "   ")
    print()
