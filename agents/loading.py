# Creating agents and finding where their saved models are.

from agents.approx_q import ApproxQAgent
from agents.dqn import DQNAgent


def model_path(kind, layout):
    if kind == "approxq":
        return "models/approxq_" + layout + ".json"
    return "models/dqn_" + layout + ".pt"


def new_agent(kind, env, episodes):
    if kind == "approxq":
        # epsilon reaches its minimum at around 60% of the training
        decay_episodes = max(1, int(episodes * 0.6))
        decay = 0.05 ** (1 / decay_episodes)
        return ApproxQAgent(eps_decay=decay)
    obs_shape = (env.n_channels, env.height, env.width)
    return DQNAgent(obs_shape, eps_decay_episodes=int(episodes * 0.6))


def load_agent(kind, env, path=None):
    if path is None:
        path = model_path(kind, env.layout_name)
    if kind == "approxq":
        agent = ApproxQAgent()
    else:
        agent = DQNAgent((env.n_channels, env.height, env.width))
    agent.load(path)
    return agent
