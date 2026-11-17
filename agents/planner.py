# A lookahead planner (expectimax). It doesn't learn anything: it knows the real
# game rules and the way the ghosts move, tries every move a few steps ahead,
# averages over where the ghosts could go and picks the move with the best
# expected score. It shows roughly how well the game can be played at all.

from pacman import env as game
from pacman.ghosts import chase_moves, move_options


class State:
    # the parts of the game that change while planning
    def __init__(self, pacman, ghosts, dirs, scared, food, capsules):
        self.pacman = pacman
        self.ghosts = ghosts
        self.dirs = dirs
        self.scared = scared
        self.food = food
        self.capsules = capsules

    def copy(self):
        # food and capsules are never changed in place, so they can be shared
        return State(self.pacman, list(self.ghosts), list(self.dirs), list(self.scared),
                     self.food, self.capsules)


class PlannerAgent:
    def __init__(self, depth=2):
        self.depth = depth

    def act(self, env, obs=None, greedy=False):
        state = State(env.pacman, list(env.ghosts), list(env.ghost_dirs), list(env.scared),
                      set(env.food), set(env.capsules))
        best_action = None
        best_value = None
        for action in env.legal_actions(state.pacman):
            value = self.expected_value(env, state, action, self.depth)
            if best_value is None or value > best_value:
                best_value = value
                best_action = action
        return best_action

    def end_episode(self):
        pass

    def best_value(self, env, state, depth):
        if depth == 0:
            return self.evaluate(env, state)
        best = None
        for action in env.legal_actions(state.pacman):
            value = self.expected_value(env, state, action, depth)
            if best is None or value > best:
                best = value
        return best

    def expected_value(self, env, state, action, depth):
        total = 0
        for prob, reward, next_state in self.outcomes(env, state, action, depth):
            if next_state is None:  # the game is over
                total += prob * reward
            else:
                total += prob * (reward + self.best_value(env, next_state, depth - 1))
        return total

    def evaluate(self, env, state):
        # at the end of the lookahead: being closer to the food is better
        closest = None
        for cell in state.food:
            distance = env.distance(state.pacman, cell)
            if closest is None or distance < closest:
                closest = distance
        return -closest

    def outcomes(self, env, state, action, depth):
        # the same order of events as PacmanEnv.step
        # returns (probability, reward, next state), next state is None when the game ended
        start = state.copy()
        start.pacman = env.next_position(start.pacman, action)
        reward = game.STEP_PENALTY
        caught, ghost_reward = self.check_ghosts(env, start)
        reward += ghost_reward
        if caught:
            return [(1.0, reward + game.LOSE_PENALTY, None)]
        if start.pacman in start.food:
            start.food = start.food - {start.pacman}
            reward += game.FOOD_REWARD
        if start.pacman in start.capsules:
            start.capsules = start.capsules - {start.pacman}
            reward += game.CAPSULE_REWARD
            start.scared = [env.scared_time] * len(start.ghosts)
        if len(start.food) == 0:
            return [(1.0, reward + game.WIN_REWARD, None)]

        results = []
        for prob, ghosts, dirs in self.ghost_outcomes(env, start, depth):
            state = start.copy()
            state.ghosts = ghosts
            state.dirs = dirs
            caught, ghost_reward = self.check_ghosts(env, state)
            if caught:
                results.append((prob, reward + ghost_reward + game.LOSE_PENALTY, None))
                continue
            for i in range(len(state.scared)):
                state.scared[i] = max(0, state.scared[i] - 1)
            results.append((prob, reward + ghost_reward, state))
        return results

    def check_ghosts(self, env, state):
        reward = 0
        for i in range(len(state.ghosts)):
            if state.ghosts[i] != state.pacman:
                continue
            if state.scared[i] == 0:
                return True, reward
            reward += game.GHOST_REWARD
            state.ghosts[i] = env.ghost_starts[i]
            state.dirs[i] = None
            state.scared[i] = 0
        return False, reward

    def ghost_outcomes(self, env, state, depth):
        # every combination of ghost moves with its probability
        combos = [(1.0, [], [])]
        for i in range(len(state.ghosts)):
            moves = self.ghost_moves(env, state, i, depth)
            new_combos = []
            for prob, ghosts, dirs in combos:
                for move_prob, position, direction in moves:
                    new_combos.append((prob * move_prob, ghosts + [position], dirs + [direction]))
            combos = new_combos
        return combos

    def ghost_moves(self, env, state, i, depth):
        # the same rules as pacman/ghosts.py, as probabilities instead of a random pick
        ghost = state.ghosts[i]
        if env.distance(ghost, state.pacman) > 2 * depth + 1:
            # too far away to matter in this lookahead, so don't branch on it
            return [(1.0, ghost, state.dirs[i])]
        options = move_options(env, ghost, state.dirs[i])
        chasing = chase_moves(env, ghost, options, state.scared[i] > 0, state.pacman)
        moves = []
        for action in options:
            prob = (1 - env.chase_prob) / len(options)
            if action in chasing:
                prob += env.chase_prob / len(chasing)
            moves.append((prob, env.next_position(ghost, action), action))
        return moves
