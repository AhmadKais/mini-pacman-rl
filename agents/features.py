# Features for approximate Q-learning: Q(s, a) = sum of w_i * f_i(s, a)
# Every feature looks at the cell pacman ends up in after taking the action.


def ghost_reach(env, ghost):
    # the cells a ghost can be in after one move
    cells = {ghost}
    for action in range(4):
        cells.add(env.next_position(ghost, action))
    return cells


def closest_distance(env, start, targets):
    best = None
    for target in targets:
        d = env.distance(start, target)
        if best is None or d < best:
            best = d
    return best


def extract_features(env, action):
    features = {"bias": 1.0}
    next_cell = env.next_position(env.pacman, action)
    maze_size = env.width * env.height

    active_ghosts = []
    scared_ghosts = []
    for i in range(len(env.ghosts)):
        if env.scared[i] > 0:
            scared_ghosts.append(env.ghosts[i])
        else:
            active_ghosts.append(env.ghosts[i])

    danger = 0
    for ghost in active_ghosts:
        if next_cell in ghost_reach(env, ghost):
            danger += 1
    features["ghosts-1-step-away"] = danger
    if len(active_ghosts) > 0:
        d = closest_distance(env, next_cell, active_ghosts)
        features["ghost-closeness"] = 1 / (d + 1)

    if danger == 0 and next_cell in env.food:
        features["eats-food"] = 1.0
    if len(env.food) > 0:
        d = closest_distance(env, next_cell, env.food)
        features["closest-food"] = d / maze_size

    if danger == 0 and next_cell in env.capsules:
        features["eats-capsule"] = 1.0

    if len(scared_ghosts) > 0:
        if next_cell in scared_ghosts:
            features["eats-scared-ghost"] = 1.0
        d = closest_distance(env, next_cell, scared_ghosts)
        features["scared-ghost-closeness"] = 1 / (d + 1)

    # dividing by 10 keeps the weight updates from blowing up
    for name in features:
        features[name] = features[name] / 10
    return features
