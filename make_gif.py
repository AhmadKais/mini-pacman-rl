# Records a trained agent playing and saves it as an animated gif,
# so the agents can be shown without running anything.
#   python make_gif.py --agent dqn --layout medium
#   python make_gif.py --agent approxq --layout small --seed 3

import argparse
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"  # draw without opening a window
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
from PIL import Image

from agents.loading import load_agent
from pacman.env import PacmanEnv
from pacman.render import PygameView


def grab(view):
    # copies the picture that pygame drew into a PIL image
    width, height = view.screen.get_size()
    pixels = pygame.image.tobytes(view.screen, "RGB")
    return Image.frombytes("RGB", (width, height), pixels)


def record_game(env, agent, view):
    frames = []
    obs = env.reset()
    action = None
    view.draw(env, action)
    frames.append(grab(view))
    while not env.done:
        action = agent.act(env, obs, greedy=True)
        obs, reward, done, info = env.step(action)
        view.draw(env, action)
        frames.append(grab(view))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["approxq", "dqn"], default="dqn")
    parser.add_argument("--layout", default="small")
    parser.add_argument("--seed", type=int, default=0, help="which ghost moves to record")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--tries", type=int, default=20, help="games to try to get a win")
    args = parser.parse_args()

    env = PacmanEnv(args.layout, max_steps=300, seed=args.seed)
    agent = load_agent(args.agent, env)
    view = PygameView(env, fps=1000)  # no waiting between frames while recording

    # record games until one is won, a gif of a lost game isn't a good demo
    frames = []
    for attempt in range(args.tries):
        frames = record_game(env, agent, view)
        if env.won:
            break
    print(f"recorded a game with {len(frames)} frames, won: {env.won}, score {env.score}")

    # hold the last frame for a second so you can see the result
    durations = [int(1000 / args.fps)] * len(frames)
    durations[-1] = 1500
    path = f"results/{args.agent}_{args.layout}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    view.close()
    print("saved", path)


if __name__ == "__main__":
    main()
