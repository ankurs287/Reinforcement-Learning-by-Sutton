import numpy as np

EPSILON_GREEDY = "EPSILON GREEDY"
UPPER_BOUND_CONFIDENCE = "UPPER BOUND CONFIDENCE"
GRADIENT_BANDIT_ALGORITHM = "GRADIENT BANDIT ALGORITHM"


class MultiArmBandit:

    def __init__(self, arms, method=EPSILON_GREEDY, epsilon=0., mu=0., var=1., confidence=0., alpha=0., baseline=False):
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
        self.average_reward = 0.

    def reset(self):
        self.action_count = np.zeros(self.arms)
        self.q_estimate = np.zeros(self.arms)
        self.q_true = self.var * np.random.randn(self.arms) + self.mu
        self.best_arm = np.argmax(self.q_true)
        self.time = 0

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
            if not self.baseline:
                rt = 0
            one_hot = np.zeros(self.arms)
            one_hot[action] = 1
            self.q_estimate[action] += self.alpha * (reward - rt)(one_hot - self.action_prob[action])
            # for arm in range(self.arms):
            #     if arm != action:
            #         self.q_estimate[arm] += -self.alpha * (reward - rt)(self.action_prob[arm])
        return reward
