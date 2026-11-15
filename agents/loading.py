# Creating agents and finding where their saved models are.

from agents.approx_q import ApproxQAgent
from agents.dqn import DQNAgent


def model_extension(kind):
    if kind == "approxq":
        return ".json"
    return ".pt"


def model_path(kind, layout):
    # where train.py saves a model when no run name is given
    return "models/" + kind + "_" + layout + model_extension(kind)


def new_agent(kind, env, episodes, double_dqn=True, drop_features=None):
    if kind == "approxq":
        # epsilon reaches its minimum at around 60% of the training
        decay_episodes = max(1, int(episodes * 0.6))
        decay = 0.05 ** (1 / decay_episodes)
        return ApproxQAgent(eps_decay=decay, drop_features=drop_features)
    obs_shape = (env.n_channels, env.height, env.width)
    return DQNAgent(obs_shape, eps_decay_episodes=int(episodes * 0.6), double_dqn=double_dqn)


def load_agent(kind, env, path=None):
    if path is None:
        path = model_path(kind, env.layout_name)
    if kind == "approxq":
        agent = ApproxQAgent()
    else:
        agent = DQNAgent((env.n_channels, env.height, env.width))
    agent.load(path)
    return agent
