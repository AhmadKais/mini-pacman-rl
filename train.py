# Train an agent on mini pacman.
#   python train.py --agent approxq --episodes 500
#   python train.py --agent dqn --episodes 3000 --live

import argparse
import csv
import os

from agents.features import extract_features
from agents.loading import model_path, new_agent
from pacman.env import PacmanEnv


def run_episode(env, agent, kind):
    obs = env.reset()
    while not env.done:
        action = agent.act(env, obs)
        if kind == "approxq":
            features = extract_features(env, action)
            next_obs, reward, done, info = env.step(action)
            # running out of time is not a real game over, so it still looks ahead
            terminal = done and not info["truncated"]
            agent.update(features, reward, env, terminal)
        else:
            next_obs, reward, done, info = env.step(action)
            terminal = done and not info["truncated"]
            agent.remember(obs, action, reward, next_obs, terminal, env.legal_mask())
        obs = next_obs
    agent.end_episode()
    return env.score, env.won


def evaluate(env, agent, games=100):
    # play without random moves and report the win rate and average score
    wins = 0
    total_score = 0
    for game in range(games):
        obs = env.reset()
        while not env.done:
            action = agent.act(env, obs, greedy=True)
            obs, reward, done, info = env.step(action)
        if env.won:
            wins += 1
        total_score += env.score
    return wins / games, total_score / games


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
          f"| epsilon {agent.epsilon:.3f}")


def save_log(path, log):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "won"])
        for row in log:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["approxq", "dqn"], default="approxq")
    parser.add_argument("--layout", default="small")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--ghost-chase", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--live", action="store_true", help="show a live chart and games while training")
    parser.add_argument("--watch-every", type=int, default=None, help="show a game every N episodes")
    args = parser.parse_args()

    episodes = args.episodes
    if episodes is None:
        if args.agent == "approxq":
            episodes = 500
        else:
            episodes = 3000

    env = PacmanEnv(args.layout, max_steps=args.max_steps, ghost_chase_prob=args.ghost_chase)
    agent = new_agent(args.agent, env, episodes)

    live = None
    if args.live:
        from pacman.live_view import LiveTraining
        watch_every = args.watch_every
        if watch_every is None:
            watch_every = max(10, episodes // 20)
        live = LiveTraining(env, args.agent + " on " + args.layout, watch_every=watch_every)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    log = []
    for episode in range(1, episodes + 1):
        score, won = run_episode(env, agent, args.agent)
        log.append((episode, score, int(won)))
        if live is not None:
            live.update(env, agent, log)
        if episode % 50 == 0:
            print_progress(log, agent)

    path = model_path(args.agent, args.layout)
    agent.save(path)
    print("saved model to", path)
    if args.agent == "approxq":
        print("learned weights:")
        for name in sorted(agent.weights):
            print(f"  {name:25s} {agent.weights[name]:9.2f}")

    save_log("results/" + args.agent + "_" + args.layout + ".csv", log)

    win_rate, avg_score = evaluate(env, agent)
    print(f"test over 100 games: win rate {win_rate:.0%}, avg score {avg_score:.1f}")
    if live is not None:
        live.close()


if __name__ == "__main__":
    main()
