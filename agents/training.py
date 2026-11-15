# Playing one training episode, and testing an agent without random moves.


def run_episode(env, agent, kind):
    obs = env.reset()
    while not env.done:
        action = agent.act(env, obs)
        if kind == "approxq":
            features = agent.get_features(env, action)
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
