# Replay buffer for DQN: stores the last transitions and gives back random batches.
# When it's full the oldest transitions get overwritten.

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.next_obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros((capacity, 4), dtype=bool)  # legal moves in the next state

    def add(self, obs, action, reward, next_obs, done, mask):
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.masks[i] = mask
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.dones[idx], self.masks[idx])
