# Turns the game state into a stack of grids for the DQN.
# Channels: walls, food, capsules, pacman, ghosts, scared ghosts (value = scared time left)
# and, if the env has ghost_directions on, the cell each ghost just came from.
# Ghosts never turn back, so that last channel tells the network where they are going.

import numpy as np

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


def make_observation(env):
    obs = np.zeros((env.n_channels, env.height, env.width), dtype=np.float32)
    obs[0] = env.walls
    for cell in env.food:
        obs[1][cell] = 1
    for cell in env.capsules:
        obs[2][cell] = 1
    obs[3][env.pacman] = 1
    for i in range(len(env.ghosts)):
        if env.scared[i] > 0:
            obs[5][env.ghosts[i]] = env.scared[i] / env.scared_time
        else:
            obs[4][env.ghosts[i]] = 1

    if env.ghost_directions:
        for i in range(len(env.ghosts)):
            direction = env.ghost_dirs[i]
            if direction is None:
                continue  # the ghost hasn't moved yet
            row = env.ghosts[i][0] - MOVES[direction][0]
            col = env.ghosts[i][1] - MOVES[direction][1]
            obs[6][row, col] = 1
    return obs
