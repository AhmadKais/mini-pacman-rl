# Mini Pac-Man environment.
# Pacman wins when all the food is eaten and loses if a ghost catches him.
# Eating a capsule scares the ghosts for a few steps so pacman can eat them.

import random

import numpy as np

from layouts import load_layout, maze_distances

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
OPPOSITE = [1, 0, 3, 2]

STEP_PENALTY = -1
FOOD_REWARD = 10
CAPSULE_REWARD = 20
GHOST_REWARD = 200
WIN_REWARD = 500
LOSE_PENALTY = -500


class PacmanEnv:
    n_actions = 4
    n_channels = 6  # walls, food, capsules, pacman, ghosts, scared ghosts

    def __init__(self, layout="small", max_steps=500, ghost_chase_prob=0.2,
                 scared_time=20, seed=None):
        (self.walls, self.start_food, self.start_capsules,
         self.pacman_start, self.ghost_starts) = load_layout(layout)
        self.height, self.width = self.walls.shape
        self.max_steps = max_steps
        self.chase_prob = ghost_chase_prob
        self.scared_time = scared_time
        self.rng = random.Random(seed)
        self.dist = maze_distances(self.walls)
        self.reset()

    def distance(self, a, b):
        return self.dist[a].get(b, 999)

    def next_position(self, pos, action):
        r, c = pos[0] + MOVES[action][0], pos[1] + MOVES[action][1]
        if self.walls[r, c]:  # layouts are surrounded by walls, so no bounds check
            return pos
        return (r, c)

    def legal_actions(self, pos=None):
        if pos is None:
            pos = self.pacman
        return [a for a in range(4) if self.next_position(pos, a) != pos]

    def legal_mask(self):
        mask = np.zeros(4, dtype=bool)
        mask[self.legal_actions()] = True
        return mask

    def reset(self):
        self.pacman = self.pacman_start
        self.ghosts = list(self.ghost_starts)
        self.ghost_dirs = [None] * len(self.ghosts)
        self.scared = [0] * len(self.ghosts)
        self.food = set(self.start_food)
        self.capsules = set(self.start_capsules)
        self.steps, self.score = 0, 0
        self.done = self.won = self.truncated = False
        return self.get_observation()

    def step(self, action):
        reward = STEP_PENALTY
        self.pacman = self.next_position(self.pacman, action)
        reward += self.check_ghosts()

        if not self.done:
            if self.pacman in self.food:
                self.food.remove(self.pacman)
                reward += FOOD_REWARD
            if self.pacman in self.capsules:
                self.capsules.remove(self.pacman)
                reward += CAPSULE_REWARD
                self.scared = [self.scared_time] * len(self.ghosts)
            if len(self.food) == 0:
                self.done = self.won = True
                reward += WIN_REWARD

        if not self.done:
            self.move_ghosts()
            reward += self.check_ghosts()

        self.scared = [max(0, s - 1) for s in self.scared]
        self.steps += 1
        if not self.done and self.steps >= self.max_steps:
            self.done = self.truncated = True  # ran out of time

        self.score += reward
        info = {"won": self.won, "truncated": self.truncated, "score": self.score}
        return self.get_observation(), reward, self.done, info

    def check_ghosts(self):
        reward = 0
        for i, g in enumerate(self.ghosts):
            if g != self.pacman:
                continue
            if self.scared[i] > 0:  # pacman eats the ghost, it goes back home
                reward += GHOST_REWARD
                self.ghosts[i] = self.ghost_starts[i]
                self.ghost_dirs[i] = None
                self.scared[i] = 0
            else:
                self.done = True
                return reward + LOSE_PENALTY
        return reward

    def move_ghosts(self):
        for i, g in enumerate(self.ghosts):
            options = self.legal_actions(g)
            # don't turn back unless it's a dead end
            if self.ghost_dirs[i] is not None and len(options) > 1:
                back = OPPOSITE[self.ghost_dirs[i]]
                if back in options:
                    options.remove(back)
            if self.rng.random() < self.chase_prob:
                d = {a: self.distance(self.next_position(g, a), self.pacman) for a in options}
                best = max(d.values()) if self.scared[i] else min(d.values())
                options = [a for a in options if d[a] == best]
            a = self.rng.choice(options)
            self.ghosts[i] = self.next_position(g, a)
            self.ghost_dirs[i] = a

    def get_observation(self):
        obs = np.zeros((6, self.height, self.width), dtype=np.float32)
        obs[0] = self.walls
        for p in self.food:
            obs[1][p] = 1
        for p in self.capsules:
            obs[2][p] = 1
        obs[3][self.pacman] = 1
        for p, s in zip(self.ghosts, self.scared):
            if s > 0:
                obs[5][p] = s / self.scared_time
            else:
                obs[4][p] = 1
        return obs
