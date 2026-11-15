# Approximate Q-learning with a linear model over hand made features.
#   Q(s,a) = w . f(s,a)
#   diff   = r + gamma * max Q(s',a') - Q(s,a)
#   w_i   += alpha * diff * f_i(s,a)

import json
import random
from collections import defaultdict

from features import extract_features


class ApproxQAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, eps_min=0.05, eps_decay=0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.weights = defaultdict(float)

    def q_value(self, feats):
        return sum(self.weights[k] * v for k, v in feats.items())

    def best_q(self, env):
        return max(self.q_value(extract_features(env, a)) for a in env.legal_actions())

    def act(self, env, obs=None, greedy=False):
        legal = env.legal_actions()
        if not greedy and random.random() < self.epsilon:
            return random.choice(legal)
        qs = {a: self.q_value(extract_features(env, a)) for a in legal}
        top = max(qs.values())
        return random.choice([a for a in legal if qs[a] == top])

    def update(self, feats, reward, env, terminal):
        # feats were computed before the step, env is already in the next state
        target = reward
        if not terminal:
            target += self.gamma * self.best_q(env)
        diff = target - self.q_value(feats)
        for k, v in feats.items():
            self.weights[k] += self.alpha * diff * v

    def end_episode(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.weights, f, indent=2)

    def load(self, path):
        with open(path) as f:
            self.weights = defaultdict(float, json.load(f))
