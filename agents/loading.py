# Creating agents and finding where their saved models are.

import torch

from agents.approx_q import ApproxQAgent
from agents.dqn import DQNAgent


def model_extension(kind):
    if kind == "approxq":
        return ".json"
    return ".pt"


def model_path(kind, layout):
    # where train.py saves a model when no run name is given
    return "models/" + kind + "_" + layout + model_extension(kind)


def new_agent(kind, env, episodes, double_dqn=True, drop_features=None, eps_episodes=None):
    # by default epsilon reaches its minimum at around 60% of the training
    if eps_episodes is None:
        eps_episodes = max(1, int(episodes * 0.6))
    if kind == "approxq":
        decay = 0.05 ** (1 / eps_episodes)
        return ApproxQAgent(eps_decay=decay, drop_features=drop_features)
    obs_shape = (env.n_channels, env.height, env.width)
    return DQNAgent(obs_shape, eps_decay_episodes=eps_episodes, double_dqn=double_dqn)


def load_agent(kind, env, path=None):
    if path is None:
        path = model_path(kind, env.layout_name)
    if kind == "approxq":
        agent = ApproxQAgent()
    else:
        # a model trained with the ghost direction channel expects 7 channels
        weights = torch.load(path, map_location="cpu")
        channels = weights["conv1.weight"].shape[1]
        env.use_ghost_directions(channels == 7)
        agent = DQNAgent((env.n_channels, env.height, env.width))
    agent.load(path)
    return agent
