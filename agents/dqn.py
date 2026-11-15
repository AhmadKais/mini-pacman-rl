# DQN agent (PyTorch). The grid is fed to a small CNN as a 6 channel image.
# Uses a replay buffer, a target network and double DQN.
# Moves into walls are masked out so the agent only picks legal actions.

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, channels, height, width, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * height * width, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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


class DQNAgent:
    def __init__(self, obs_shape, n_actions=4, lr=2.5e-4, gamma=0.99, batch_size=64,
                 buffer_size=100000, warmup=2000, train_every=2, target_every=1000,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=1500, reward_scale=0.01):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.train_every = train_every
        self.target_every = target_every
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.reward_scale = reward_scale  # rewards go up to 500, so scale them down
        self.steps = 0
        self.episodes = 0

        channels, height, width = obs_shape
        self.net = QNetwork(channels, height, width, n_actions).to(self.device)
        self.target = QNetwork(channels, height, width, n_actions).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_shape)

    @property
    def epsilon(self):
        # goes down linearly per episode (early games are very short, so per step is too slow)
        progress = min(1.0, self.episodes / self.eps_decay_episodes)
        return self.eps_start + progress * (self.eps_end - self.eps_start)

    def act(self, env, obs, greedy=False):
        legal = env.legal_actions()
        if not greedy and random.random() < self.epsilon:
            return random.choice(legal)
        with torch.no_grad():
            x = torch.tensor(obs, device=self.device).unsqueeze(0)
            q = self.net(x)[0].cpu().numpy()
        best = legal[0]
        for action in legal:
            if q[action] > q[best]:
                best = action
        return best

    def remember(self, obs, action, reward, next_obs, terminal, next_mask):
        self.buffer.add(obs, action, reward * self.reward_scale, next_obs, terminal, next_mask)
        self.steps += 1
        if self.steps >= self.warmup and self.steps % self.train_every == 0:
            self.train_step()
        if self.steps % self.target_every == 0:
            self.target.load_state_dict(self.net.state_dict())

    def train_step(self):
        obs, actions, rewards, next_obs, dones, masks = self.buffer.sample(self.batch_size)
        obs = torch.tensor(obs, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_obs = torch.tensor(next_obs, device=self.device)
        dones = torch.tensor(dones, device=self.device)
        masks = torch.tensor(masks, device=self.device)

        q = self.net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # double DQN: the online net picks the next action, the target net scores it
            next_q = self.net(next_obs).masked_fill(~masks, -1e9)
            best_next = next_q.argmax(1, keepdim=True)
            next_value = self.target(next_obs).gather(1, best_next).squeeze(1)
            target = rewards + self.gamma * (1 - dones) * next_value

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.optimizer.step()

    def end_episode(self):
        self.episodes += 1

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.net.state_dict())
