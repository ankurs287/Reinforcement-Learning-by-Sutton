import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

EPSILON_GREEDY = "EPSILON GREEDY"
UPPER_BOUND_CONFIDENCE = "UPPER BOUND CONFIDENCE"
GRADIENT_BANDIT_ALGORITHM = "GRADIENT BANDIT ALGORITHM"


class MultiArmBandit:

    def __init__(self, arms, method=EPSILON_GREEDY, epsilon=0., mu=0, var=1, confidence=0, alpha=0., baseline=False):
        self.arms = arms
        self.arms_index = np.arange(self.arms)
        self.epsilon = epsilon
        self.var = var
        self.c = confidence
        self.method = method
        self.alpha = alpha
        self.baseline = baseline
        self.mu = mu
        self.time = 0
        self.average_reward = 0

    def reset(self):
        self.action_count = np.zeros(self.arms)
        self.q_estimate = np.zeros(self.arms)
        self.q_true = self.var * np.random.randn(self.arms) + self.mu
        self.best_arm = np.argmax(self.q_true)

    def act(self):
        if self.method == EPSILON_GREEDY:
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.arms)  # exploring
            return np.argmax(self.q_estimate)  # being greedy
        elif self.method == UPPER_BOUND_CONFIDENCE:
            UCB_estimation = self.q_estimate + \
                             self.c * np.sqrt(np.log(self.time) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice([action for action, q in enumerate(UCB_estimation) if q == q_best])
        elif self.method == GRADIENT_BANDIT_ALGORITHM:
            exp_est = np.exp(self.q_estimate)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.arms_index, p=self.action_prob)

    def step(self, action):
        self.action_count[action] += 1
        reward = (self.var * np.random.randn()) + self.q_true[action]
        self.average_reward = (self.time - 1.0) / self.time * self.average_reward + reward / self.time
        if self.method == EPSILON_GREEDY or UPPER_BOUND_CONFIDENCE:
            self.q_estimate[action] += 1.0 / self.action_count[action] * (reward - self.q_estimate[action])
        elif self.method == GRADIENT_BANDIT_ALGORITHM:
            rt = self.average_reward
            if self.baseline:
                rt = 0
            self.q_estimate[action] += self.alpha * (reward - rt)(1 - self.action_prob[action])
            for arm in range(self.arms):
                if arm != action:
                    self.q_estimate[arm] += -self.alpha * (reward - rt)(self.action_prob[arm])
        return reward

    def simulate(self, runs, time):
        abs_estimation_error = np.zeros((self.arms, runs, time))
        self.best_action_counts = np.zeros((runs, time))
        self.rewards = np.zeros((runs, time))
        for r in tqdm(range(runs)):
            self.reset()
            for t in range(time):
                self.time += 1
                action = self.act()
                if action == self.best_arm:
                    self.best_action_counts[r, t] = 1
                reward = self.step(action)
                self.rewards[r, t] = reward
                # for arm in range(self.arms):
                # abs_estimation_error[arm, r, t] = abs(self.q_estimate[arm] - self.q_true[arm])
        self.rewards = self.rewards.mean(axis=0)  # taking average of all the runs
        self.best_action_counts = self.best_action_counts.mean(axis=0)  # taking average of all the runs
        abs_estimation_error = abs_estimation_error.mean(axis=1)  # taking average of all the runs
        return self.rewards, self.best_action_counts, abs_estimation_error


def q1_q2(var=1):
    plt.figure(figsize=(20, 30))

    runs = 2000
    time = 1000
    arms = 10

    bandit0 = MultiArmBandit(arms, epsilon=0, var=var)
    bandit1 = MultiArmBandit(arms, epsilon=0.01, var=var)
    bandit2 = MultiArmBandit(arms, epsilon=0.1, var=var)

    rewards0, best_action_counts0, abs_estimation_error0 = bandit0.simulate(runs, time)
    rewards1, best_action_counts1, abs_estimation_error1 = bandit1.simulate(runs, time)
    rewards2, best_action_counts2, abs_estimation_error2 = bandit2.simulate(runs, time)

    # average reward vs steps
    plt.subplot(3, 2, 1)
    plt.xlabel('steps')
    plt.ylabel('average reward')

    plt.plot(rewards0, label='epsilon = 0')
    plt.plot(rewards1, label='epsilon = 0.01')
    plt.plot(rewards2, label='epsilon = 0.1')
    plt.legend()

    # optimal action vs steps
    plt.subplot(3, 2, 2)
    plt.xlabel('steps')
    plt.ylabel('% optimal action')

    plt.plot(best_action_counts0, label='epsilon = 0')
    plt.plot(best_action_counts1, label='epsilon = 0.01')
    plt.plot(best_action_counts2, label='epsilon = 0.1')
    plt.legend()

    # average absolute error in the estimate vs steps for epsilon = 0
    plt.subplot(3, 2, 3)
    plt.xlabel('steps')
    plt.ylabel('average absolute error in the estimate for epsilson = 0')
    for arm in range(arms):
        plt.plot(abs_estimation_error0[arm], label='arm = %s' % arm)
    plt.legend()

    # average absolute error in the estimate vs steps for epsilon = 0.01
    plt.subplot(3, 2, 4)
    plt.xlabel('steps')
    plt.ylabel('average absolute error in the estimate for epsilson = 0.01')
    for arm in range(arms):
        plt.plot(abs_estimation_error1[arm], label='arm = %s' % arm)
    plt.legend()

    # average absolute error in the estimate vs steps for epsilon = 0.1
    plt.subplot(3, 2, 5)
    plt.xlabel('steps')
    plt.ylabel('average absolute error in the estimate for epsilson = 0.1')
    for arm in range(arms):
        plt.plot(abs_estimation_error2[arm], label='arm = %s' % arm)
    plt.legend()

    if var == 1:
        plt.savefig('./q1.png')
    else:
        plt.savefig('./q2.png')
    plt.close()


def q6():
    plt.figure(figsize=(10, 30))

    runs = 2000
    time = 1000
    arms = 10

    bandit0 = MultiArmBandit(arms, epsilon=0.1)
    bandit1 = MultiArmBandit(arms, method=UPPER_BOUND_CONFIDENCE, confidence=2)
    bandit2 = MultiArmBandit(arms, method=UPPER_BOUND_CONFIDENCE, confidence=1)
    bandit3 = MultiArmBandit(arms, method=UPPER_BOUND_CONFIDENCE, confidence=4)

    rewards0, _, _ = bandit0.simulate(runs, time)
    rewards1, _, _ = bandit1.simulate(runs, time)
    rewards2, _, _ = bandit2.simulate(runs, time)
    rewards3, _, _ = bandit3.simulate(runs, time)

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

    plt.savefig('./q6.png')


def q7():
    runs = 2000
    time = 1000
    arms = 10


    bandit0 = MultiArmBandit(arms, mu=4, method=GRADIENT_BANDIT_ALGORITHM, alpha=0.1, baseline=True)
    bandit1 = MultiArmBandit(arms, mu=4, method=GRADIENT_BANDIT_ALGORITHM, alpha=0.1, baseline=False)
    bandit2 = MultiArmBandit(arms, mu=4, method=GRADIENT_BANDIT_ALGORITHM, alpha=0.4, baseline=True)
    bandit3 = MultiArmBandit(arms, mu=4, method=GRADIENT_BANDIT_ALGORITHM, alpha=0.4, baseline=False)

    _, best_action_counts0, _ = bandit0.simulate(runs, time)
    _, best_action_counts1, _ = bandit1.simulate(runs, time)
    _, best_action_counts2, _ = bandit2.simulate(runs, time)
    _, best_action_counts3, _ = bandit3.simulate(runs, time)


    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')

    plt.plot(best_action_counts0, label='alpha = 0.1 w/ baseline')
    plt.plot(best_action_counts1, label='alpha = 0.1 w/o baseline')
    plt.plot(best_action_counts2, label='alpha = 0.4 w/ baseline')
    plt.plot(best_action_counts3, label='alpha = 0.4 w/0 baseline')

    plt.legend()

    plt.savefig('./q7.png')
    plt.close()


# print("Solving Q1..")
# q1_q2()
#
# print("Solving Q2..")
# q1_q2(var=2)

# print("Solving Q6..")
# q6()

# print("Solving Q7..")
# q7()
