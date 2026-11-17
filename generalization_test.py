# Can an agent play a maze it was never trained on?
# Approximate Q-learning's features don't depend on the maze, so its weights can be
# used on any maze. DQN can't: its network only fits the size of its training maze.
#   python generalization_test.py --games 200

import argparse
import json

from agents.loading import load_agent
from agents.training import evaluate
from pacman.env import PacmanEnv

MODELS = ["models/approxq_small.json", "models/approxq_medium.json"]
MAZES = ["tiny", "small", "medium", "unseen"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    args = parser.parse_args()

    results = []
    print(f"{'weights':28s} {'maze':8s} {'win':>5s} {'score':>7s}")
    for model in MODELS:
        for maze in MAZES:
            env = PacmanEnv(maze, max_steps=300, seed=0)
            agent = load_agent("approxq", env, model)
            win_rate, avg_score = evaluate(env, agent, games=args.games)
            print(f"{model:28s} {maze:8s} {win_rate:5.0%} {avg_score:7.1f}")
            results.append({"weights": model, "maze": maze,
                            "win_rate": win_rate, "avg_score": avg_score})

    with open("results/generalization.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved results/generalization.json")


if __name__ == "__main__":
    main()
