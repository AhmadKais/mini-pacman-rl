# Mini pacman game environment.
# Pacman wins when all the food is eaten and loses if a ghost catches him.
# Eating a capsule scares the ghosts for a few steps so pacman can eat them.

import random

import numpy as np

from pacman.ghosts import move_ghosts
from pacman.layouts import read_layout, maze_distances
from pacman.observation import make_observation

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
ACTION_NAMES = ["Up", "Down", "Left", "Right"]

STEP_PENALTY = -1
FOOD_REWARD = 10
CAPSULE_REWARD = 20
GHOST_REWARD = 200
WIN_REWARD = 500
LOSE_PENALTY = -500


class PacmanEnv:
    n_actions = 4

    def __init__(self, layout="small", max_steps=500, ghost_chase_prob=0.2,
                 scared_time=20, seed=None, ghost_directions=False):
        walls, food, capsules, pacman, ghosts = read_layout(layout)
        self.layout_name = layout
        self.walls = walls
        self.start_food = food
        self.start_capsules = capsules
        self.pacman_start = pacman
        self.ghost_starts = ghosts
        self.height = walls.shape[0]
        self.width = walls.shape[1]
        self.max_steps = max_steps
        self.chase_prob = ghost_chase_prob
        self.scared_time = scared_time
        self.rng = random.Random(seed)
        self.dist = maze_distances(walls)
        self.use_ghost_directions(ghost_directions)
        self.reset()

    def distance(self, a, b):
        if b in self.dist[a]:
            return self.dist[a][b]
        return 999

    def next_position(self, pos, action):
        row = pos[0] + MOVES[action][0]
        col = pos[1] + MOVES[action][1]
        if self.walls[row, col]:
            return pos  # walking into a wall means staying in place
        return (row, col)

    def legal_actions(self, pos=None):
        if pos is None:
            pos = self.pacman
        actions = []
        for action in range(4):
            if self.next_position(pos, action) != pos:
                actions.append(action)
        return actions

    def legal_mask(self):
        mask = np.zeros(4, dtype=bool)
        for action in self.legal_actions():
            mask[action] = True
        return mask

    def reset(self):
        self.pacman = self.pacman_start
        self.ghosts = list(self.ghost_starts)
        self.ghost_dirs = [None] * len(self.ghosts)
        self.scared = [0] * len(self.ghosts)
        self.food = set(self.start_food)
        self.capsules = set(self.start_capsules)
        self.steps = 0
        self.score = 0
        self.ghosts_eaten = 0
        self.done = False
        self.won = False
        self.truncated = False
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
                self.done = True
                self.won = True
                reward += WIN_REWARD

        if not self.done:
            move_ghosts(self)
            reward += self.check_ghosts()

        for i in range(len(self.scared)):
            self.scared[i] = max(0, self.scared[i] - 1)
        self.steps += 1
        if not self.done and self.steps >= self.max_steps:
            self.done = True
            self.truncated = True  # ran out of time

        self.score += reward
        info = {"won": self.won, "truncated": self.truncated, "score": self.score}
        return self.get_observation(), reward, self.done, info

    def check_ghosts(self):
        reward = 0
        for i in range(len(self.ghosts)):
            if self.ghosts[i] != self.pacman:
                continue
            if self.scared[i] > 0:
                # pacman eats the ghost and it goes back to its start
                reward += GHOST_REWARD
                self.ghosts_eaten += 1
                self.ghosts[i] = self.ghost_starts[i]
                self.ghost_dirs[i] = None
                self.scared[i] = 0
            else:
                self.done = True
                return reward + LOSE_PENALTY
        return reward

    def use_ghost_directions(self, on):
        # adds a 7th observation channel with the cell each ghost just came from
        self.ghost_directions = on
        if on:
            self.n_channels = 7
        else:
            self.n_channels = 6

    def get_observation(self):
        return make_observation(self)
