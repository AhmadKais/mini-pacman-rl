# Does the agent really avoid ghosts, or does it just run for the food?
# Compares the trained agents with two simple baselines:
#   random     - picks any legal move
#   food-only  - always walks toward the closest food and ignores the ghosts
# A "close call" is when an active ghost gets within 2 steps of pacman.
# We count it as escaped if pacman is still alive 3 steps later.
#
#   python ghost_test.py --games 200

import argparse
import random

from pacman_env import PacmanEnv
from play import load_agent


class RandomAgent:
    def act(self, env, obs, greedy=True):
        return random.choice(env.legal_actions())


class FoodOnlyAgent:
    def act(self, env, obs, greedy=True):
        def food_dist(a):
            nxt = env.next_position(env.pacman, a)
            return min(env.distance(nxt, f) for f in env.food)
        return min(env.legal_actions(), key=food_dist)


def ghost_near(env):
    return any(s == 0 and env.distance(env.pacman, g) <= 2
               for g, s in zip(env.ghosts, env.scared))


def run(env, agent, games):
    wins = caught = timeouts = 0
    close_calls = escapes = ghosts_eaten = 0
    for _ in range(games):
        obs = env.reset()
        danger_steps = []  # steps where a close call started
        while not env.done:
            if ghost_near(env) and (not danger_steps or env.steps - danger_steps[-1] > 3):
                danger_steps.append(env.steps)
            obs, reward, _, _ = env.step(agent.act(env, obs, greedy=True))
            if reward >= 199:  # +200 for eating a scared ghost
                ghosts_eaten += 1
        end = env.steps
        died = env.done and not env.won and not env.truncated
        for t in danger_steps:
            close_calls += 1
            if not died or end - t > 3:
                escapes += 1
        wins += env.won
        caught += died
        timeouts += env.truncated
    return {
        "win": wins / games, "caught": caught / games, "timeout": timeouts / games,
        "close calls/game": close_calls / games,
        "escaped": escapes / max(1, close_calls),
        "ghosts eaten/game": ghosts_eaten / games,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="small")
    p.add_argument("--games", type=int, default=200)
    args = p.parse_args()

    for chase in [0.2, 0.5, 0.8]:
        env = PacmanEnv(args.layout, max_steps=300, ghost_chase_prob=chase)
        agents = {"random": RandomAgent(), "food-only": FoodOnlyAgent()}
        for name in ["approxq", "dqn"]:
            ext = "json" if name == "approxq" else "pt"
            try:
                agents[name] = load_agent(name, env, f"models/{name}_{args.layout}.{ext}")
            except FileNotFoundError:
                print(f"(no trained {name} model found, skipping)")

        print(f"\nghost chase probability {chase}  ({args.games} games each)")
        print(f"{'agent':10s} {'win':>6s} {'caught':>7s} {'timeout':>8s} "
              f"{'close/game':>11s} {'escaped':>8s} {'eaten/game':>11s}")
        for name, agent in agents.items():
            r = run(env, agent, args.games)
            print(f"{name:10s} {r['win']:6.0%} {r['caught']:7.0%} {r['timeout']:8.0%} "
                  f"{r['close calls/game']:11.1f} {r['escaped']:8.0%} "
                  f"{r['ghosts eaten/game']:11.2f}")


if __name__ == "__main__":
    main()
