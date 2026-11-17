# Tests the lookahead planner, to see how well the game can be played at all.
#   python planner_test.py --layout medium --depth 2 --games 200

import argparse
import json
import time

from agents.planner import PlannerAgent
from pacman.env import PacmanEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="medium")
    parser.add_argument("--depth", type=int, default=2, help="how many moves to look ahead")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = PacmanEnv(args.layout, max_steps=300, ghost_chase_prob=0.2, seed=args.seed)
    agent = PlannerAgent(depth=args.depth)
    wins = 0
    caught = 0
    timeouts = 0
    total_score = 0
    start = time.time()
    for game in range(1, args.games + 1):
        env.reset()
        while not env.done:
            env.step(agent.act(env))
        total_score += env.score
        if env.won:
            wins += 1
        elif env.truncated:
            timeouts += 1
        else:
            caught += 1
        if game % 20 == 0:
            print(f"{game} games: win rate {wins / game:.0%}, "
                  f"{(time.time() - start) / game:.1f} s per game", flush=True)

    result = {"layout": args.layout, "depth": args.depth, "games": args.games,
              "win_rate": wins / args.games, "caught": caught / args.games,
              "timeout": timeouts / args.games, "avg_score": total_score / args.games}
    print(result)
    path = f"results/planner_{args.layout}_depth{args.depth}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
