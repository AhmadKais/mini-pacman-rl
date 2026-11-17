# Train an agent on mini pacman.
#   python train.py --agent approxq --episodes 500
#   python train.py --agent dqn --episodes 3000 --live
#   python train.py --agent dqn --seed 1 --name experiments/dqn_small_seed1

import argparse
import csv
import json
import os
import random

import numpy as np
import torch

from agents.loading import model_extension, new_agent
from agents.training import evaluate, keep_if_best, run_episode
from pacman.env import PacmanEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["approxq", "dqn"], default="approxq")
    parser.add_argument("--layout", default="small")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--ghost-chase", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", default=None, help="run name, used for the output files")
    parser.add_argument("--no-double", action="store_true", help="DQN without double DQN")
    parser.add_argument("--drop-features", nargs="*", default=None,
                        help="approximate Q-learning features to leave out")
    parser.add_argument("--eps-episodes", type=int, default=None,
                        help="episodes until epsilon is at its minimum (default 60%% of them)")
    parser.add_argument("--eps-end", type=float, default=0.05, help="lowest epsilon (DQN)")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="test every N episodes and keep the best model (0 = keep the last)")
    parser.add_argument("--ghost-directions", action="store_true",
                        help="DQN also sees which way the ghosts are moving")
    parser.add_argument("--threads", type=int, default=None, help="cpu threads for torch")
    parser.add_argument("--live", action="store_true", help="show a live chart and games")
    parser.add_argument("--watch-every", type=int, default=None, help="show a game every N episodes")
    parser.add_argument("--fps", type=int, default=15, help="speed of the games in --live")
    return parser.parse_args()


def print_progress(log, agent):
    last = log[-50:]
    total_score = 0
    wins = 0
    for episode, score, won in last:
        total_score += score
        wins += won
    avg_score = total_score / len(last)
    win_rate = wins / len(last)
    print(f"episode {len(log):5d} | avg score {avg_score:7.1f} | win rate {win_rate:5.0%} "
          f"| epsilon {agent.epsilon:.3f}", flush=True)


def save_log(path, log):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "won"])
        for row in log:
            writer.writerow(row)


def main():
    args = parse_args()
    episodes = args.episodes
    if episodes is None:
        if args.agent == "approxq":
            episodes = 500
        else:
            episodes = 3000
    name = args.name
    if name is None:
        name = args.agent + "_" + args.layout
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    env = PacmanEnv(args.layout, max_steps=args.max_steps,
                    ghost_chase_prob=args.ghost_chase, seed=args.seed,
                    ghost_directions=args.ghost_directions)
    agent = new_agent(args.agent, env, episodes, double_dqn=not args.no_double,
                      drop_features=args.drop_features, eps_episodes=args.eps_episodes,
                      eps_end=args.eps_end)
    model_file = "models/" + name + model_extension(args.agent)
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    live = None
    if args.live:
        from pacman.live_view import LiveTraining
        watch_every = args.watch_every
        if watch_every is None:
            watch_every = max(10, episodes // 20)
        live = LiveTraining(env, args.agent + " on " + args.layout, watch_every=watch_every,
                            fps=args.fps)

    log = []
    best = None  # (win rate, avg score) of the best model so far
    for episode in range(1, episodes + 1):
        score, won = run_episode(env, agent, args.agent)
        log.append((episode, score, int(won)))
        if live is not None:
            live.update(env, agent, log)
        if episode % 50 == 0:
            print_progress(log, agent)
        if args.eval_every > 0 and episode % args.eval_every == 0:
            best = keep_if_best(env, agent, model_file, best)

    if best is not None:
        agent.load(model_file)  # go back to the best model for the final test
    else:
        agent.save(model_file)
    print("saved model to", model_file)
    if args.agent == "approxq":
        print("learned weights:")
        for feature in sorted(agent.weights):
            print(f"  {feature:25s} {agent.weights[feature]:9.2f}")

    os.makedirs(os.path.dirname("results/" + name), exist_ok=True)
    save_log("results/" + name + ".csv", log)

    win_rate, avg_score = evaluate(env, agent)
    print(f"test over 100 games: win rate {win_rate:.0%}, avg score {avg_score:.1f}")
    with open("results/" + name + "_test.json", "w") as f:
        json.dump({"win_rate": win_rate, "avg_score": avg_score, "seed": args.seed,
                   "episodes": episodes}, f, indent=2)
    if live is not None:
        live.close()


if __name__ == "__main__":
    main()
