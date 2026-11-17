# Tests for the DQN agent and the replay buffer.
#   python -m unittest -v

import os
import tempfile
import unittest

import numpy as np
import torch

from agents.dqn import DQNAgent, QNetwork
from agents.replay_buffer import ReplayBuffer
from pacman.env import PacmanEnv

RIGHT = 3

# pacman can only go right from the start
CORRIDOR = """
%%%%%%
%P..o%
%%%%%%
"""


def small_agent(env):
    shape = (env.n_channels, env.height, env.width)
    return DQNAgent(shape, buffer_size=100, warmup=10, batch_size=8, eps_decay_episodes=10)


class TestNetwork(unittest.TestCase):
    def test_one_q_value_per_action(self):
        net = QNetwork(6, 7, 9, 4)
        q = net(torch.zeros(2, 6, 7, 9))
        self.assertEqual(tuple(q.shape), (2, 4))


class TestReplayBuffer(unittest.TestCase):
    def test_overwrites_the_oldest_when_full(self):
        buffer = ReplayBuffer(3, (1, 2, 2))
        for i in range(5):
            obs = np.full((1, 2, 2), i, dtype=np.float32)
            buffer.add(obs, 0, i, obs, False, np.ones(4, dtype=bool))
        self.assertEqual(buffer.size, 3)
        self.assertEqual(sorted(buffer.rewards.tolist()), [2.0, 3.0, 4.0])


class TestDQNAgent(unittest.TestCase):
    def test_only_picks_legal_moves(self):
        env = PacmanEnv(CORRIDOR)
        agent = small_agent(env)
        obs = env.get_observation()
        for i in range(20):
            self.assertEqual(agent.act(env, obs), RIGHT)  # random moves
        self.assertEqual(agent.act(env, obs, greedy=True), RIGHT)

    def test_epsilon_goes_down_per_episode(self):
        env = PacmanEnv(CORRIDOR)
        agent = small_agent(env)
        self.assertAlmostEqual(agent.epsilon, 1.0)
        for i in range(5):
            agent.end_episode()
        self.assertAlmostEqual(agent.epsilon, 0.525)  # halfway between 1.0 and 0.05
        for i in range(10):
            agent.end_episode()
        self.assertAlmostEqual(agent.epsilon, 0.05)

    def test_training_changes_the_network(self):
        env = PacmanEnv(CORRIDOR)
        agent = small_agent(env)
        before = agent.net.fc2.weight.detach().clone()
        for i in range(20):
            obs = env.reset()
            next_obs, reward, done, info = env.step(RIGHT)
            agent.remember(obs, RIGHT, reward, next_obs, done, env.legal_mask())
        self.assertFalse(torch.equal(before, agent.net.fc2.weight))

    def test_save_and_load(self):
        env = PacmanEnv(CORRIDOR)
        agent = small_agent(env)
        path = os.path.join(tempfile.mkdtemp(), "model.pt")
        agent.save(path)
        other = small_agent(env)
        other.load(path)
        obs = torch.tensor(env.get_observation()).unsqueeze(0)
        self.assertTrue(torch.equal(agent.net(obs), other.net(obs)))


if __name__ == "__main__":
    unittest.main()
