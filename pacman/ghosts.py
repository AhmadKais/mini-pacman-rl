# How the ghosts move.
# Mostly random, but sometimes a ghost chases pacman (or runs away when it is scared).

OPPOSITE = [1, 0, 3, 2]  # the reverse of up, down, left, right


def move_ghosts(env):
    for i in range(len(env.ghosts)):
        ghost = env.ghosts[i]
        options = move_options(env, ghost, env.ghost_dirs[i])
        if env.rng.random() < env.chase_prob:
            options = chase_moves(env, ghost, options, env.scared[i] > 0, env.pacman)
        action = env.rng.choice(options)
        env.ghosts[i] = env.next_position(ghost, action)
        env.ghost_dirs[i] = action


def move_options(env, ghost, direction):
    # ghosts don't turn back unless they are in a dead end
    options = env.legal_actions(ghost)
    if direction is not None and len(options) > 1:
        back = OPPOSITE[direction]
        if back in options:
            options.remove(back)
    return options


def chase_moves(env, ghost, options, scared, pacman):
    # keep the moves that get closest to pacman, or furthest away if the ghost is scared
    dists = {}
    for action in options:
        dists[action] = env.distance(env.next_position(ghost, action), pacman)
    if scared:
        best = max(dists.values())
    else:
        best = min(dists.values())
    moves = []
    for action in options:
        if dists[action] == best:
            moves.append(action)
    return moves
