# Two simple agents to compare the learned ones against.

import random


class RandomAgent:
    # picks any legal move
    def act(self, env, obs=None, greedy=True):
        return random.choice(env.legal_actions())


class FoodOnlyAgent:
    # always walks toward the closest food and ignores the ghosts
    def act(self, env, obs=None, greedy=True):
        best_action = None
        best_dist = None
        for action in env.legal_actions():
            next_cell = env.next_position(env.pacman, action)
            dist = None
            for food in env.food:
                d = env.distance(next_cell, food)
                if dist is None or d < dist:
                    dist = d
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_action = action
        return best_action
