# Watch a trained agent play.
#   python play.py --agent approxq
#   python play.py --agent dqn --games 5 --text

import argparse
import os
import time

from pacman_env import PacmanEnv
from render import PygameView, text_frame


def load_agent(kind, env, path):
    if kind == "approxq":
        from approx_q_agent import ApproxQAgent
        agent = ApproxQAgent()
    else:
        from dqn_agent import DQNAgent
        agent = DQNAgent((env.n_channels, env.height, env.width))
    agent.load(path)
    return agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["approxq", "dqn"], default="approxq")
    p.add_argument("--layout", default="small")
    p.add_argument("--model", default=None)
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--ghost-chase", type=float, default=0.2)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--text", action="store_true", help="print to terminal instead of pygame")
    args = p.parse_args()

    env = PacmanEnv(args.layout, max_steps=300, ghost_chase_prob=args.ghost_chase)
    ext = "json" if args.agent == "approxq" else "pt"
    path = args.model or f"models/{args.agent}_{args.layout}.{ext}"
    agent = load_agent(args.agent, env, path)
    view = None if args.text else PygameView(env, fps=args.fps)

    wins = 0
    for game in range(1, args.games + 1):
        obs = env.reset()
        action = None
        while True:
            if view:
                view.draw(env, action)
                if view.closed:
                    return
            else:
                os.system("cls" if os.name == "nt" else "clear")
                print(text_frame(env))
                time.sleep(1 / args.fps)
            if env.done:
                break
            action = agent.act(env, obs, greedy=True)
            obs, _, _, _ = env.step(action)
        wins += env.won
        result = "won" if env.won else ("time up" if env.truncated else "lost")
        print(f"game {game}: {result}, score {env.score}")
        time.sleep(1)

    print(f"won {wins}/{args.games} games")
    if view:
        view.close()


if __name__ == "__main__":
    main()
