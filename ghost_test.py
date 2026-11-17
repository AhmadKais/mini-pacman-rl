# Does the agent really avoid ghosts, or does it just run for the food?
# Compares the trained agents with two baselines (see agents/baselines.py).
# A "close call" is when an active ghost gets within 2 steps of pacman.
# It counts as escaped if pacman is still alive 3 steps later.
#
#   python ghost_test.py --games 200

import argparse
import os
import random

from agents.baselines import FoodOnlyAgent, RandomAgent
from agents.loading import load_agent, model_path
from pacman.env import PacmanEnv


def ghost_is_near(env):
    for i in range(len(env.ghosts)):
        if env.scared[i] == 0 and env.distance(env.pacman, env.ghosts[i]) <= 2:
            return True
    return False


def test_agent(env, agent, games):
    stats = {"win": 0, "caught": 0, "timeout": 0, "close calls": 0,
             "escaped": 0, "ghosts eaten": 0}
    for game in range(games):
        obs = env.reset()
        close_calls = []  # the step where each close call started
        while not env.done:
            if ghost_is_near(env):
                if len(close_calls) == 0 or env.steps - close_calls[-1] > 3:
                    close_calls.append(env.steps)
            action = agent.act(env, obs, greedy=True)
            obs, reward, done, info = env.step(action)

        stats["ghosts eaten"] += env.ghosts_eaten
        caught = not env.won and not env.truncated
        for start in close_calls:
            stats["close calls"] += 1
            if not caught or env.steps - start > 3:
                stats["escaped"] += 1
        if env.won:
            stats["win"] += 1
        elif caught:
            stats["caught"] += 1
        else:
            stats["timeout"] += 1
    return stats


def print_row(name, stats, games):
    close_calls = stats["close calls"]
    escaped = stats["escaped"] / max(1, close_calls)
    print(f"{name:10s} {stats['win'] / games:6.0%} {stats['caught'] / games:7.0%} "
          f"{stats['timeout'] / games:8.0%} {close_calls / games:11.1f} {escaped:8.0%} "
          f"{stats['ghosts eaten'] / games:11.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="small")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for chase in [0.2, 0.5, 0.8]:
        env = PacmanEnv(args.layout, max_steps=300, ghost_chase_prob=chase)
        agents = {"random": RandomAgent(), "food-only": FoodOnlyAgent()}
        for kind in ["approxq", "dqn"]:
            path = model_path(kind, args.layout)
            if os.path.exists(path):
                agents[kind] = load_agent(kind, env, path)
            else:
                print(f"(no trained {kind} model at {path}, skipping it)")

        print(f"\nghosts chase {chase:.0%} of the time ({args.games} games each)")
        print(f"{'agent':10s} {'win':>6s} {'caught':>7s} {'timeout':>8s} "
              f"{'close/game':>11s} {'escaped':>8s} {'eaten/game':>11s}")
        for name in agents:
            # every agent gets the same random numbers, so the results can be repeated
            random.seed(args.seed)
            env.rng = random.Random(args.seed)
            stats = test_agent(env, agents[name], args.games)
            print_row(name, stats, args.games)


if __name__ == "__main__":
    main()
