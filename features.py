# Features for approximate Q-learning: Q(s,a) = sum(w_i * f_i(s,a))
# Every feature looks at where pacman ends up after taking the action.


def ghost_reach(env, pos):
    # cells a ghost at pos can be in after one move
    return {env.next_position(pos, a) for a in range(4)} | {pos}


def extract_features(env, action):
    f = {"bias": 1.0}
    nxt = env.next_position(env.pacman, action)
    size = env.width * env.height

    active = [g for g, s in zip(env.ghosts, env.scared) if s == 0]
    scared = [g for g, s in zip(env.ghosts, env.scared) if s > 0]

    danger = sum(nxt in ghost_reach(env, g) for g in active)
    f["ghosts-1-step-away"] = danger
    if active:
        d = min(env.distance(nxt, g) for g in active)
        f["ghost-closeness"] = 1 / (d + 1)

    if not danger and nxt in env.food:
        f["eats-food"] = 1.0
    if env.food:
        d = min(env.distance(nxt, food) for food in env.food)
        f["closest-food"] = d / size

    if not danger and nxt in env.capsules:
        f["eats-capsule"] = 1.0

    if scared:
        if nxt in scared:
            f["eats-scared-ghost"] = 1.0
        d = min(env.distance(nxt, g) for g in scared)
        f["scared-ghost-closeness"] = 1 / (d + 1)

    # dividing by 10 keeps the weight updates from blowing up
    return {k: v / 10 for k, v in f.items()}
