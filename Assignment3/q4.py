
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


HIT = 'Hit'
STICK = 'Stick'
actions = [STICK, HIT]
value = np.zeros((10, 10, 2))
TENK = 10000
FIVEL = 500000
TRUE_VALUE = -0.27726

class State:
    def __init__(self, player_usable_ace, player_card_sum, dealer_face_up_card):
        self.player_usable_ace = player_usable_ace
        self.dealer_face_up_card = dealer_face_up_card
        self.player_card_sum = player_card_sum


class Env:
    def __init__(self, state=None, action=None):
        self.state = state
        self.action = action
        if state is not None:
            self.set_dealer_face_down_card()

    @staticmethod
    def equi_probable_random_state_action_pair():
        env = Env()
        player_card_sum = np.random.randint(12, 21 + 1)
        dealer_face_up_card = np.random.randint(1, 10 + 1)
        player_usable_ace = bool(np.random.randint(2))
        env.state = State(player_usable_ace, player_card_sum, dealer_face_up_card)
        env.action = np.random.choice(actions)
        env.set_dealer_face_down_card()
        return env

    def set_dealer_face_down_card(self):
        dealer_face_down_card = hit()
        dealer_card_sum = 0
        dealer_usable_ace = True
        if is_ace(self.state.dealer_face_up_card) and is_ace(dealer_face_down_card):
            dealer_card_sum = 12
        elif is_ace(self.state.dealer_face_up_card):
            dealer_card_sum = 11 + dealer_face_down_card
        elif is_ace(dealer_face_down_card):
            dealer_card_sum = 11 + self.state.dealer_face_up_card
        else:
            self.state.dealer_face_up_card + dealer_face_down_card
            dealer_usable_ace = False
        self.dealer_usable_ace = dealer_usable_ace
        self.dealer_card_sum = dealer_card_sum


# In[3]:


def is_face_card(card):
    return True if card >= 11 else False


def is_ace(card):
    return True if card == 1 else False


# In[4]:


# draws a card from a standard 52-card deck with replacement and return card's value
def hit():
    # randomly draw a card from a suit (size = 13). 11, 12, 13 represents 3 Face Cards
    card = np.random.randint(1, 13 + 1)
    if is_face_card(card):
        return 10
    else:
        return card


# In[5]:


def target_policy(player_card_sum):
    if player_card_sum >= 20:
        return STICK
    return HIT


def behavior_policy(player_card_sum):
    return np.random.choice(actions)


# In[6]:


def init_env():
    # init player's card sum (12-21) [i.e, sum of all the cards player holds]
    player_card_sum = 0
    player_ace_count = 0
    player_usable_ace = False

    while player_card_sum < 12:
        card = hit()

        if is_ace(card):
            player_ace_count += 1
            if player_card_sum + 11 <= 21:
                player_usable_ace = True
                player_card_sum += 11
            else:
                player_card_sum += 1

        else:
            player_card_sum += card

    # init dealer's state
    dealer_face_up_card = hit()
    
    initial_state = State(player_usable_ace, player_card_sum, dealer_face_up_card)
    return Env(initial_state)


# In[7]:


def play(env, policy=target_policy):
    # Generate an episode following pi from initial state
    episode = []

    player_card_sum = env.state.player_card_sum
    player_usable_ace = env.state.player_usable_ace
    dealer_face_up_card = env.state.dealer_face_up_card
    dealer_card_sum = env.dealer_card_sum
    dealer_usable_ace = env.dealer_usable_ace

    # Player Act until he sticks or goes bust
    action = policy(player_card_sum) if env.action is None else env.action
    while True:
        state = State(player_usable_ace, player_card_sum, dealer_face_up_card)
        episode.append([state, action])
        if action == HIT:
            card = hit()
            player_card_sum += card
        else:
            break

        if player_card_sum > 21 and player_usable_ace:
            player_card_sum -= 10
            player_usable_ace = False
        elif player_card_sum > 21:
            return -1, episode
        action = policy(player_card_sum)

    # Dealer Act
    while True:
        if dealer_card_sum < 17:
            card = hit()
            if is_ace(card) and dealer_card_sum + 11 < 22:
                dealer_card_sum += 11
                dealer_usable_ace = True
            else:
                dealer_card_sum += card
        else:
            break

        if dealer_card_sum > 21 and dealer_usable_ace:
            dealer_card_sum -= 10
            dealer_usable_ace = False
        elif dealer_card_sum > 21:
            return 1, episode

    if player_card_sum > dealer_card_sum:
        return 1, episode
    elif player_card_sum == dealer_card_sum:
        return 0, episode
    return -1, episode


# ### Monte Carlo

# In[8]:


def every_visit_mc(n_episodes):
    state_value = np.zeros((2, 10, 10))
    count_usable = np.ones(state_value.shape)
    for i in range(n_episodes):
        # Choose initial state(player cards sum, dealer's face-up card, usable or non-usable ace) randomly
        env = init_env()
        g, episode = play(env)
        T = len(episode)
        for t in range(T - 1, -1, -1):
            state, action = episode[t]
            player_card_sum = state.player_card_sum - 12
            dealer_face_up_card = state.dealer_face_up_card - 1
            player_usable_ace = state.player_usable_ace
            u = 1 if player_usable_ace else 0
            state_value[u, player_card_sum, dealer_face_up_card] += g
            count_usable[u, player_card_sum, dealer_face_up_card] += 1
    
    state_value = state_value / count_usable
    return state_value


# In[9]:


state_value_tk = every_visit_mc(TENK)


# In[10]:


state_value_fl = every_visit_mc(FIVEL)


# In[11]:


print("Usable Ace, Episodes: ", TENK)
print(np.round(state_value_tk[1], 1))
print('-' * 20)
print("No Usable Ace, Episodes: ", TENK)
print(np.round(state_value_tk[0], 1))
print("\n" * 2)

print("Usable Ace, Episodes: ", FIVEL)
print(np.round(state_value_fl[1], 1))
print('-' * 20)
print("No Usable Ace, Episodes: ", FIVEL)
print(np.round(state_value_fl[0], 1))


# ### Monte Carlo Exploring Starts

# In[12]:


def es_mc():
    state_action_value = np.zeros((2, 10, 10, 2))  # usable ace x player card sum x dealer's face up card x actions
    count = np.ones(state_action_value.shape)
    EPISODES = FIVEL
    for i in range(EPISODES):
        # Choose initial state(player cards sum, dealer's face-up card, usable or non-usable ace) randomly
        env = Env.equi_probable_random_state_action_pair()
        g, episode = play(env)
        T = len(episode)
        for t in range(T - 1, -1, -1):
            state, action = episode[t]
            player_card_sum = state.player_card_sum - 12
            dealer_face_up_card = state.dealer_face_up_card - 1
            player_usable_ace = state.player_usable_ace
            u = 1 if player_usable_ace else 0
            state_action_value[u, player_card_sum, dealer_face_up_card, actions.index(action)] += g
            count[u, player_card_sum, dealer_face_up_card, actions.index(action)] += 1

    state_action_value = state_action_value / count
    return state_action_value


# In[13]:


state_action_value = es_mc()


# In[14]:


print('Usable Ace Optimal Value Function')
print(np.round(np.max(state_action_value[1], axis=-1), 1))
print('-' * 100)
print('No Usable Ace Optimal Value Function')
print(np.round(np.max(state_action_value[0], axis=-1), 1))
print('\n' * 2)

print('1: HIT, 0: STICK')
print('Usable Ace Optimal Policy')
print(np.argmax(state_action_value[1], axis=-1))
print('-' * 100)
print('No Usable Ace Optimal Policy')
print(np.argmax(state_action_value[0], axis=-1))
print('1: HIT, 0: STICK')


# ### Off-Policy MC 

# In[15]:


def off_policy_mc(n_episodes):
    rhos = np.zeros(n_episodes)
    returns = np.zeros(n_episodes)
    for i in range(n_episodes):
        # Choose initial state(player cards sum, dealer's face-up card, usable or non-usable ace) randomly
        state = State(player_usable_ace=True, player_card_sum=13, dealer_face_up_card=2)
        env = Env(state=state)
        g, episode = play(env, behavior_policy)
        T = len(episode)
        rho = 1.0
        for t in range(T - 1, -1, -1):
            state, action = episode[t]
            if action == target_policy(state.player_card_sum):
                rho *= 2
            else:
                rho = 0.0
                break
        rhos[i] = rho
        returns[i] = g
    return np.add.accumulate(rhos * returns), np.add.accumulate(rhos)


# In[16]:


episodes = TENK
runs = 100
error_ordinary = np.zeros((runs, episodes))
error_weighted = np.zeros((runs, episodes))
from tqdm import tqdm

for i in tqdm(range(0, runs)):
    weighted_returns_sum, rhos_sum = off_policy_mc(episodes)
    ordinary_sampling = weighted_returns_sum / np.arange(1, episodes + 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        weighted_sampling = np.where(rhos_sum != 0, weighted_returns_sum / rhos_sum, 0)

    error_ordinary[i] = np.power(ordinary_sampling - TRUE_VALUE, 2)
    error_weighted[i] = np.power(weighted_sampling - TRUE_VALUE, 2)


# In[17]:


plt.plot(np.mean(error_ordinary, axis=0), label='Ordinary Importance Sampling')
plt.plot(np.mean(error_weighted, axis=0), label='Weighted Importance Sampling')
plt.xlabel('Episodes (log scale)')
plt.ylabel('Mean square error')
plt.xscale('log')
plt.legend()
plt.savefig('q4_3.png')
plt.close()

