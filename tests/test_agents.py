# Tests for the features and the agents.
#   python -m unittest -v

import os
import tempfile
import unittest

from agents.approx_q import ApproxQAgent
from agents.baselines import FoodOnlyAgent
from agents.features import extract_features
from pacman.env import PacmanEnv

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

GHOST_NEARBY = """
%%%%%%%
%P...G%
%%%%%%%
"""

FOOD_BOTH_WAYS = """
%%%%%%%
%..P .%
%%%%%%%
"""


class TestFeatures(unittest.TestCase):
    def test_ghost_one_step_away(self):
        env = PacmanEnv(GHOST_NEARBY)
        env.pacman = (1, 3)
        towards = extract_features(env, RIGHT)
        away = extract_features(env, LEFT)
        self.assertGreater(towards["ghosts-1-step-away"], 0)
        self.assertEqual(away["ghosts-1-step-away"], 0)

    def test_no_eats_food_when_ghost_is_close(self):
        env = PacmanEnv(GHOST_NEARBY)
        env.pacman = (1, 3)
        features = extract_features(env, RIGHT)  # food there, but next to the ghost
        self.assertNotIn("eats-food", features)

    def test_eats_food(self):
        env = PacmanEnv(FOOD_BOTH_WAYS)
        features = extract_features(env, LEFT)
        self.assertIn("eats-food", features)


class TestApproxQ(unittest.TestCase):
    def test_update_moves_weights_toward_reward(self):
        env = PacmanEnv(FOOD_BOTH_WAYS)
        agent = ApproxQAgent(alpha=0.5)
        features = agent.get_features(env, LEFT)
        env.step(LEFT)
        agent.update(features, 10, env, terminal=True)
        self.assertGreater(agent.q_value(features), 0)

    def test_dropped_features_are_not_used(self):
        env = PacmanEnv(GHOST_NEARBY)
        agent = ApproxQAgent(drop_features=["ghosts-1-step-away", "ghost-closeness"])
        features = agent.get_features(env, RIGHT)
        self.assertNotIn("ghosts-1-step-away", features)
        self.assertNotIn("ghost-closeness", features)

    def test_save_and_load(self):
        agent = ApproxQAgent()
        agent.weights = {"bias": 1.5, "eats-food": 20.0}
        path = os.path.join(tempfile.mkdtemp(), "weights.json")
        agent.save(path)
        other = ApproxQAgent()
        other.load(path)
        self.assertEqual(other.weights, agent.weights)


class TestBaselines(unittest.TestCase):
    def test_food_only_goes_to_closest_food(self):
        env = PacmanEnv(FOOD_BOTH_WAYS)
        env.food = {(1, 5)}
        self.assertEqual(FoodOnlyAgent().act(env), RIGHT)


if __name__ == "__main__":
    unittest.main()
