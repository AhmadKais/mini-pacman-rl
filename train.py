# Train an agent on mini pacman.
#   python train.py --agent approxq --episodes 500
#   python train.py --agent dqn --episodes 3000

import argparse
import csv
import os

from features import extract_features
from pacman_env import PacmanEnv


def make_agent(name, env, episodes):
    if name == "approxq":
        from approx_q_agent import ApproxQAgent
        # decay epsilon so it gets to the minimum around 60% of training
        decay = 0.05 ** (1 / max(1, int(episodes * 0.6)))
        return ApproxQAgent(eps_decay=decay)
    from dqn_agent import DQNAgent
    obs_shape = (env.n_channels, env.height, env.width)
    return DQNAgent(obs_shape, eps_decay_episodes=int(episodes * 0.6))


def run_episode(env, agent, kind):
    obs = env.reset()
    while not env.done:
        action = agent.act(env, obs)
        if kind == "approxq":
            feats = extract_features(env, action)
        next_obs, reward, done, info = env.step(action)
        terminal = done and not info["truncated"]  # time limit is not a real end
        if kind == "approxq":
            agent.update(feats, reward, env, terminal)
        else:
            agent.remember(obs, action, reward, next_obs, terminal, env.legal_mask())
        obs = next_obs
    agent.end_episode()
    return env.score, env.won


def evaluate(env, agent, n=100):
    wins, total = 0, 0
    for _ in range(n):
        obs = env.reset()
        while not env.done:
            obs, _, _, _ = env.step(agent.act(env, obs, greedy=True))
        wins += env.won
        total += env.score
    return wins / n, total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["approxq", "dqn"], default="approxq")
    p.add_argument("--layout", default="small")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--ghost-chase", type=float, default=0.2)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--live", action="store_true", help="show live chart + games while training")
    p.add_argument("--watch-every", type=int, default=None, help="show a game every N episodes")
    args = p.parse_args()

    episodes = args.episodes or (500 if args.agent == "approxq" else 3000)
    env = PacmanEnv(args.layout, max_steps=args.max_steps, ghost_chase_prob=args.ghost_chase)
    agent = make_agent(args.agent, env, episodes)
    live = None
    if args.live:
        from live import LiveTraining
        every = args.watch_every or max(10, episodes // 20)
        live = LiveTraining(env, f"{args.agent} on {args.layout}", watch_every=every)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    log = []
    for ep in range(1, episodes + 1):
        score, won = run_episode(env, agent, args.agent)
        log.append((ep, score, int(won)))
        if live:
            live.update(env, agent, log)
        if ep % 50 == 0:
            last = log[-50:]
            avg = sum(x[1] for x in last) / len(last)
            wins = sum(x[2] for x in last) / len(last)
            print(f"episode {ep:5d} | avg score {avg:7.1f} | win rate {wins:5.0%} "
                  f"| epsilon {agent.epsilon:.3f}")

    ext = "json" if args.agent == "approxq" else "pt"
    model_path = f"models/{args.agent}_{args.layout}.{ext}"
    agent.save(model_path)
    print("saved model to", model_path)
    if args.agent == "approxq":
        print("learned weights:")
        for k, v in sorted(agent.weights.items()):
            print(f"  {k:25s} {v:9.2f}")

    csv_path = f"results/{args.agent}_{args.layout}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "score", "won"])
        w.writerows(log)

    win_rate, avg = evaluate(env, agent)
    print(f"greedy test over 100 games: win rate {win_rate:.0%}, avg score {avg:.1f}")
    if live:
        live.close()


if __name__ == "__main__":
    main()
