# Tests for the game rules. Run from the project folder with:
#   python -m unittest -v

import unittest

from pacman import env as game
from pacman.env import PacmanEnv

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# small corridors so we know exactly what will happen
NO_GHOSTS = """
%%%%%%
%P..o%
%%%%%%
"""

ONE_FOOD = """
%%%%
%P.%
%%%%
"""

GHOST_AHEAD = """
%%%%%%
%P..G%
%%%%%%
"""

CAPSULE_THEN_GHOST = """
%%%%%%%
%Po..G%
%%%%%%%
"""


class TestMoving(unittest.TestCase):
    def test_walking_into_a_wall_stays_in_place(self):
        env = PacmanEnv(NO_GHOSTS)
        start = env.pacman
        obs, reward, done, info = env.step(UP)
        self.assertEqual(env.pacman, start)
        self.assertEqual(reward, game.STEP_PENALTY)

    def test_legal_actions_in_a_corridor(self):
        env = PacmanEnv(NO_GHOSTS)
        self.assertEqual(env.legal_actions(), [RIGHT])

    def test_maze_distance(self):
        env = PacmanEnv(NO_GHOSTS)
        self.assertEqual(env.distance((1, 1), (1, 4)), 3)
        self.assertEqual(env.distance((1, 4), (1, 4)), 0)


class TestEating(unittest.TestCase):
    def test_eating_food(self):
        env = PacmanEnv(NO_GHOSTS)
        obs, reward, done, info = env.step(RIGHT)
        self.assertEqual(reward, game.STEP_PENALTY + game.FOOD_REWARD)
        self.assertEqual(len(env.food), 1)
        self.assertFalse(done)

    def test_eating_the_last_food_wins(self):
        env = PacmanEnv(ONE_FOOD)
        obs, reward, done, info = env.step(RIGHT)
        self.assertTrue(done)
        self.assertTrue(info["won"])
        self.assertEqual(reward, game.STEP_PENALTY + game.FOOD_REWARD + game.WIN_REWARD)


class TestGhosts(unittest.TestCase):
    def test_ghost_catches_pacman(self):
        env = PacmanEnv(GHOST_AHEAD)
        env.step(RIGHT)  # the ghost can only come toward pacman in this corridor
        self.assertEqual(env.ghosts[0], (1, 3))
        obs, reward, done, info = env.step(RIGHT)
        self.assertTrue(done)
        self.assertFalse(info["won"])
        self.assertEqual(reward, game.STEP_PENALTY + game.LOSE_PENALTY)

    def test_capsule_scares_the_ghosts(self):
        env = PacmanEnv(CAPSULE_THEN_GHOST, scared_time=20)
        obs, reward, done, info = env.step(RIGHT)
        self.assertEqual(reward, game.STEP_PENALTY + game.CAPSULE_REWARD)
        self.assertEqual(env.scared[0], 19)  # one step of the timer already used

    def test_eating_a_scared_ghost(self):
        env = PacmanEnv(CAPSULE_THEN_GHOST)
        env.step(RIGHT)  # capsule
        obs, reward, done, info = env.step(RIGHT)  # food, then the ghost walks into pacman
        self.assertEqual(reward, game.STEP_PENALTY + game.FOOD_REWARD + game.GHOST_REWARD)
        self.assertFalse(done)
        self.assertEqual(env.ghosts[0], env.ghost_starts[0])  # sent back home
        self.assertEqual(env.scared[0], 0)


class TestEpisode(unittest.TestCase):
    def test_time_limit(self):
        env = PacmanEnv(NO_GHOSTS, max_steps=2)
        env.step(UP)
        obs, reward, done, info = env.step(UP)
        self.assertTrue(done)
        self.assertTrue(info["truncated"])
        self.assertFalse(info["won"])

    def test_reset_puts_everything_back(self):
        env = PacmanEnv(NO_GHOSTS)
        env.step(RIGHT)
        env.reset()
        self.assertEqual(env.pacman, env.pacman_start)
        self.assertEqual(len(env.food), 2)
        self.assertEqual(env.score, 0)

    def test_observation(self):
        env = PacmanEnv("small")
        obs = env.get_observation()
        self.assertEqual(obs.shape, (6, env.height, env.width))
        self.assertEqual(obs[3].sum(), 1)  # exactly one pacman
        self.assertEqual(obs[4].sum(), 1)  # one ghost, not scared
        self.assertEqual(obs[1].sum(), len(env.food))


if __name__ == "__main__":
    unittest.main()
