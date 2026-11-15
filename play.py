# Watch a trained agent play.
#   python play.py --agent approxq
#   python play.py --agent dqn --games 5 --text

import argparse
import os
import time

from agents.loading import load_agent
from pacman.env import PacmanEnv
from pacman.render import PygameView, text_frame


def show(env, view, action, fps):
    if view is not None:
        view.draw(env, action)
    else:
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
        print(text_frame(env))
        time.sleep(1 / fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["approxq", "dqn"], default="approxq")
    parser.add_argument("--layout", default="small")
    parser.add_argument("--model", default=None, help="path of the saved model")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--ghost-chase", type=float, default=0.2)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--text", action="store_true", help="show the game in the terminal")
    args = parser.parse_args()

    env = PacmanEnv(args.layout, max_steps=300, ghost_chase_prob=args.ghost_chase)
    agent = load_agent(args.agent, env, args.model)

    view = None
    if not args.text:
        view = PygameView(env, fps=args.fps)

    wins = 0
    for game in range(1, args.games + 1):
        obs = env.reset()
        action = None
        show(env, view, action, args.fps)
        while not env.done:
            action = agent.act(env, obs, greedy=True)
            obs, reward, done, info = env.step(action)
            show(env, view, action, args.fps)
            if view is not None and view.closed:
                return

        if env.won:
            wins += 1
            result = "won"
        elif env.truncated:
            result = "time up"
        else:
            result = "lost"
        print(f"game {game}: {result}, score {env.score}")
        time.sleep(1)

    print(f"won {wins}/{args.games} games")
    if view is not None:
        view.close()


if __name__ == "__main__":
    main()
