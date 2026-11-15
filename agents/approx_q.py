# Approximate Q-learning with a linear model over hand made features.
#   Q(s, a) = w . f(s, a)
#   diff    = r + gamma * max Q(s', a') - Q(s, a)
#   w_i    += alpha * diff * f_i(s, a)

import json
import random

from agents.features import extract_features


class ApproxQAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, eps_min=0.05, eps_decay=0.995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.weights = {}

    def q_value(self, features):
        total = 0.0
        for name, value in features.items():
            total += self.weights.get(name, 0.0) * value
        return total

    def best_actions(self, env):
        best_q = None
        best = []
        for action in env.legal_actions():
            q = self.q_value(extract_features(env, action))
            if best_q is None or q > best_q:
                best_q = q
                best = [action]
            elif q == best_q:
                best.append(action)
        return best, best_q

    def act(self, env, obs=None, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.choice(env.legal_actions())
        best, best_q = self.best_actions(env)
        return random.choice(best)

    def update(self, features, reward, env, terminal):
        # features were computed before the step, env is already in the next state
        target = reward
        if not terminal:
            best, best_q = self.best_actions(env)
            target += self.gamma * best_q
        diff = target - self.q_value(features)
        for name, value in features.items():
            old = self.weights.get(name, 0.0)
            self.weights[name] = old + self.alpha * diff * value

    def end_episode(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.weights, f, indent=2)

    def load(self, path):
        with open(path) as f:
            self.weights = json.load(f)
