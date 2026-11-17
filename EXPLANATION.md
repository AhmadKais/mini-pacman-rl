# Mini Pac-Man RL: the full explanation

This document explains everything in the project: the RL theory you need, the
game, both algorithms, every file and function, every experiment, and the
questions you are likely to get. Read it from top to bottom once, then use the
Q&A section (part 10) to practise.

Contents

1. The project in one minute
2. The RL theory you need
3. The game (environment)
4. Algorithm 1: approximate Q-learning
5. Algorithm 2: DQN
6. The agents that don't learn: baselines and the planner
7. Every file and every function
8. How the experiments were run and measured
9. All the results and what we learned from them
10. Questions you may be asked (with answers)
11. Honest limitations
12. How to give a live demo

---

## 1. The project in one minute

> "We built a small Pac-Man game and taught an agent to win it with two
> reinforcement learning algorithms. The first is **approximate Q-learning**: a
> linear model over 8 features I designed (like *is a ghost one step away*). The
> second is **DQN**: a convolutional neural network that looks at the raw grid and
> learns its own features. On the small maze (1 ghost) both win almost always:
> approximate Q-learning 96% after about 230 games, DQN 100% after about 2,100
> games. On the medium maze (2 ghosts) approximate Q-learning reaches 63% and DQN
> about 80%. Every result is the mean of 3 seeds. Ablations show the ghost features
> are what make approximate Q-learning work, and double DQN helps a little. A
> lookahead planner that knows the rules wins 95% on the medium maze, so there is
> still room for DQN to improve: it plays about as well as looking 1-2 moves ahead."

---

## 2. The RL theory you need

### 2.1 The setting: an MDP

Reinforcement learning is learning by trial and error. The **agent** (Pac-Man)
looks at the **state** `s`, picks an **action** `a`, the **environment** (the
game) answers with a **reward** `r` and a new state `s'`. This repeats until the
game ends (an **episode**). Formally this is a Markov Decision Process (MDP):

- states S: everything about the game right now (positions, food left, scared timers)
- actions A: up, down, left, right
- transition P(s' | s, a): where the game goes next. It's **random** here because the
  ghosts move randomly.
- reward R(s, a): the numbers in section 3.2
- discount factor γ (gamma), between 0 and 1

"Markov" means the next state only depends on the current state and action, not on
the history. (One subtle point: the direction a ghost is moving is part of the real
state, because ghosts don't turn back. Our first DQN version didn't see it, see 5.7.)

### 2.2 The goal: maximize the return

The agent wants to maximize the **return**, the discounted sum of future rewards:

```
G = r1 + γ·r2 + γ²·r3 + ...
```

γ < 1 means a reward now is worth more than the same reward later. With γ = 0.9 a
reward 10 steps away counts 0.9^10 ≈ 0.35 as much; with γ = 0.99 it counts 0.90.

### 2.3 The Q-function

`Q(s, a)` = the expected return if I take action `a` in state `s` and play well
afterwards. If we know Q, playing is easy: in every state pick `argmax_a Q(s, a)`.
This is called the **greedy policy**.

The best Q-function satisfies the **Bellman optimality equation**:

```
Q*(s, a) = E[ r + γ · max_a' Q*(s', a') ]
```

In words: the value of a move = the reward I get now + the (discounted) value of the
best move from where I land. The expectation E is over the randomness (the ghosts).

### 2.4 Q-learning

Q-learning turns the Bellman equation into an update rule. After each step
(s, a, r, s'):

```
target = r + γ · max_a' Q(s', a')          (just r if the game ended)
Q(s, a) ← Q(s, a) + α · (target − Q(s, a))
```

`target − Q(s, a)` is the **TD error** (temporal difference error): how wrong my
guess was. α (alpha) is the learning rate.

Important properties:
- **Off-policy**: the target uses `max`, the best action, not the action I will
  actually take next. So it learns the greedy policy even while exploring randomly.
  (SARSA is the on-policy version: it uses the action actually taken next.)
- **Model-free**: it never learns or uses the rules of the game, only samples.
- **Bootstrapping**: the target uses its own current estimate Q(s', a').

### 2.5 Why not a table?

Classic Q-learning stores one number per (state, action) in a table. Here the state
includes which food is left. The medium maze has 50 food dots, so there are up to
2^50 ≈ 10^15 food combinations, times all positions of Pac-Man and 2 ghosts and the
timers. A table is impossible, and it would never generalize: every new state would
start from zero. So we **approximate** Q with a function that has a fixed number of
parameters: a linear model (algorithm 1) or a neural network (algorithm 2).

### 2.6 Exploration: ε-greedy

If the agent always takes the move it currently thinks is best, it never discovers
better ones. **ε-greedy**: with probability ε pick a random legal move, otherwise
the greedy one. ε starts at 1.0 (all random) and goes down during training to a
small value (0.05, or 0.01 in the last experiment). In the **test** we use ε = 0
(no random moves), which is why test win rates are higher than training curves.

### 2.7 The "deadly triad"

Function approximation + bootstrapping + off-policy learning together can make
Q-learning unstable or even diverge (Sutton & Barto call it the deadly triad). Both
our algorithms have all three. That's why DQN needs its stabilizing tricks (replay
buffer, target network, Huber loss, gradient clipping), and why approximate
Q-learning divides its features by 10 (smaller steps).

---

## 3. The game (environment)

### 3.1 Rules

- The maze is a grid. `%` wall, `.` food, `o` capsule, `P` Pac-Man, `G` ghost.
- Pac-Man **wins** when all food is eaten (capsules don't have to be eaten).
- A ghost on the same cell as Pac-Man = **game over** (unless the ghost is scared).
- A capsule makes all ghosts **scared for 20 steps**. Pac-Man can eat a scared ghost;
  it goes back to its start position and is not scared anymore.
- A game is **cut off after 300 steps** (in all our scripts). That is called
  **truncation**, it is not a loss, see 3.5.
- Mazes (in `pacman/layouts.py`):
  - `tiny`: 7×5, 1 ghost, for quick tests
  - `small`: 9×7, 1 ghost, 23 food, 2 capsules
  - `medium`: 13×9, 2 ghosts, 50 food, 2 capsules
  - `unseen`: 11×9, 2 ghosts, never used for training, only to test generalization

### 3.2 Rewards

| event | reward | why |
|---|---|---|
| every step | −1 | so wasting time is bad, it should finish fast |
| eat food | +10 | the main goal |
| eat capsule | +20 | small bonus, capsules are useful |
| eat scared ghost | +200 | big bonus, makes hunting ghosts worth it |
| win (last food) | +500 | the real goal |
| caught | −500 | the real danger |

The **score** shown everywhere is simply the sum of all rewards in one game
(undiscounted). A typical won game on the small maze: 23 food × 10 + 500 for winning
− about 60 steps + maybe a ghost or two ≈ 900-1100.

### 3.3 How the ghosts move (`pacman/ghosts.py`)

Each step, each ghost:
1. lists its legal moves and removes the move that goes back where it came from
   (unless it's in a dead end). So ghosts walk along corridors and don't jitter.
2. with probability `chase_prob` (0.2 by default) keeps only the moves that bring it
   **closest** to Pac-Man (maze distance), or **furthest** if it is scared.
3. picks one of the remaining moves at random.

So ghosts are 80% random walkers and 20% chasers. In `ghost_test.py` we also test
against 50% and 80% chasers.

### 3.4 The order of events in one step (`PacmanEnv.step`)

This order matters and students might ask about it:

1. reward starts at −1
2. Pac-Man moves (walking into a wall = staying in place)
3. collision check: if he walked into a ghost, either he eats it (scared) or he dies
4. if alive: eat food / capsule; if no food left → win (+500)
5. if the game isn't over: **ghosts move**, then collision check again
6. scared timers go down by 1
7. step counter +1; at 300 steps → truncated

Note: Pac-Man and a ghost can't "swap places through each other" without a collision,
because Pac-Man moves first; if he steps onto the ghost's cell he is caught in step 3.

### 3.5 Truncation vs. game over

When the 300 step limit ends the game, that's not because of anything in the state;
the game was just cut off. If we treated it as a real end, the agent would learn
"the value of this state is 0", which is wrong. So in `agents/training.py`:

```python
terminal = done and not info["truncated"]
```

On a real end (win / caught) the target is just `r`. On truncation we still add
`γ · max Q(s', a')`. In practice our trained agents almost never hit the limit
(0 timeouts in 600 test games), but it's the correct way to do it.

### 3.6 What the agents see

- **Approximate Q-learning** doesn't look at the raw state, it asks the env for
  features (section 4.2).
- **DQN** gets an **observation**: a stack of grids the size of the maze, one grid
  ("channel") per kind of object (section 5.2).

---

## 4. Algorithm 1: approximate Q-learning

### 4.1 The idea

Instead of a table, Q is a weighted sum of features:

```
Q(s, a) = w1·f1(s, a) + w2·f2(s, a) + ... = w · f(s, a)
```

A feature is a number describing the situation **after** taking action `a`, e.g.
"how many ghosts could be on the cell I'm moving to". The agent learns one weight per
feature. Only 8 weights, which is why it learns in a few hundred games, and why we
can read what it learned.

### 4.2 The 8 features (`agents/features.py`)

All are computed for `next_cell` = the cell Pac-Man ends up in after action `a`.

| feature | value | meaning |
|---|---|---|
| `bias` | 1 | a constant, like the intercept in linear regression |
| `ghosts-1-step-away` | number of active ghosts that could reach `next_cell` in one move | **danger** |
| `ghost-closeness` | 1 / (distance to closest active ghost + 1) | bigger when a ghost is closer |
| `eats-food` | 1 if `next_cell` has food **and there's no danger** | eating is good, but not next to a ghost |
| `closest-food` | distance to closest food / maze size | smaller = closer to food |
| `eats-capsule` | 1 if `next_cell` has a capsule and no danger | |
| `eats-scared-ghost` | 1 if a scared ghost is on `next_cell` | |
| `scared-ghost-closeness` | 1 / (distance to closest scared ghost + 1) | encourages hunting |

Details worth knowing:
- Distances are **maze distances** (shortest path with BFS, through corridors), not
  straight-line distances. They are computed once for all pairs of cells when the env
  is created (`maze_distances`).
- `ghost_reach` = the ghost's cell plus every cell it can step to. If `next_cell` is
  in there, the ghost might be on it after the ghosts move → danger.
- `eats-food` is switched off when there's danger, so the agent never thinks
  "food is worth risking death for". Same for capsules.
- **All features are divided by 10.** Big feature values give big updates
  (`α · diff · f`) and the weights can blow up. Dividing keeps the updates small.

### 4.3 The update rule and where it comes from

After each step:

```
diff = (r + γ · max_a' Q(s', a')) − Q(s, a)
w_i ← w_i + α · diff · f_i(s, a)          for every feature i
```

Where does this come from? We want Q(s, a) to be close to the target. Take the
squared error `L = ½ (target − Q(s, a))²` and do one step of **gradient descent** on
the weights (treating the target as a constant). Because Q is linear,
`∂Q/∂w_i = f_i(s, a)`, so `−∂L/∂w_i = diff · f_i`. That's exactly the update. So
approximate Q-learning is "gradient descent on the TD error", with a linear model.

Intuition: if the outcome was **better** than expected (diff > 0), raise the weights
of the features that were active; if it was worse, lower them. When Pac-Man dies
right after moving next to a ghost, diff is very negative and `ghosts-1-step-away`
was active → its weight goes down a lot.

### 4.4 Settings

| setting | value | why |
|---|---|---|
| α (learning rate) | 0.1 | standard, stable with the /10 features |
| γ (discount) | 0.9 | short horizon is enough, features already "see" far (closest food) |
| ε | 1.0 → 0.05 | multiplied by `eps_decay` after each game |
| `eps_decay` | `0.05 ** (1 / (0.6 · episodes))` | so ε reaches exactly 0.05 at 60% of training |
| episodes | 500 (small), 1000 (medium) | |

Ties between equally good actions are broken randomly (`random.choice(best)`), so it
doesn't get stuck always going the same way.

### 4.5 What it actually learned (real weights, `models/approxq_small.json`)

| feature | weight (small) | weight (medium) |
|---|---|---|
| bias | 931 | 421 |
| ghosts-1-step-away | **−654** | **−1354** |
| ghost-closeness | −312 | −1004 |
| eats-food | 643 | 587 |
| closest-food | −14 | −23 |
| eats-capsule | 322 | 326 |
| scared-ghost-closeness | 420 | 647 |
| eats-scared-ghost | 126 | 543 |

(Remember each feature is divided by 10, so `ghosts-1-step-away` = 1 ghost
contributes −654 × 0.1 = −65 to Q.)

How to read it: danger has a big negative weight, food and capsules positive, being
close to food is good (negative weight × distance). On the medium maze, with 2
ghosts, the ghost weights became about twice as negative: it learned to be more
careful. You can show this table to explain "the model is interpretable".

### 4.6 Strengths and weaknesses

- \+ very fast (≈230 games ≈ under a minute on the small maze), interpretable, stable
- \+ the features don't depend on the maze, so the same weights work on other mazes
  with no loss (tested, see 9.6). DQN can't do this.
- − only as good as the features. On the medium maze it stops at 63%: the features
  only look at the closest ghost and the closest food, one step ahead, so it can't
  plan around two ghosts. It doesn't "know" about getting trapped in a corridor.

---

## 5. Algorithm 2: DQN (Deep Q-Network)

### 5.1 The idea

Same Q-learning target, but Q(s, ·) is a **neural network** that takes the whole
grid as input and outputs 4 numbers, one Q-value per action. No hand made features:
the network has to discover "a ghost next to me is bad" by itself. That's why it
needs many more games, but it can also find things we didn't think of (it learned to
hunt scared ghosts better than approximate Q-learning did).

DQN is from Mnih et al. 2015 (Atari). We use the same main ideas on a smaller scale.

### 5.2 The input: grids (`pacman/observation.py`)

The observation is an array of shape `(channels, height, width)`, like an image with
several color channels:

| channel | content |
|---|---|
| 0 | walls (1 = wall) |
| 1 | food |
| 2 | capsules |
| 3 | Pac-Man |
| 4 | active ghosts |
| 5 | scared ghosts, value = scared time left / 20 (so it also knows when it will end) |
| 6 | (only with `--ghost-directions`) the cell each ghost just came from |

Channel 6 was added later (section 9.4). Ghosts never turn back, so "where it came
from" + "where it is" tells the network which way it's going.

### 5.3 The network (`QNetwork` in `agents/dqn.py`)

```
input (C × H × W)
→ conv 3×3, 32 filters, padding 1, ReLU
→ conv 3×3, 64 filters, padding 1, ReLU
→ flatten (64 · H · W numbers)
→ fully connected 256, ReLU
→ fully connected 4  (one Q-value per action)
```

- **Why convolutions?** A 3×3 filter looks at a cell and its neighbours, so it can
  detect local patterns ("ghost next to Pac-Man", "food in this corridor") anywhere
  on the board with the same weights. Two layers → each output sees a 5×5 area.
- **Padding 1** keeps the grid the same size, so no border cells are lost.
- **No pooling**: the grid is tiny (9×13) and exact positions matter.
- The fully connected layer combines everything over the whole board.
- Size: about 1.9 million parameters on the medium maze, almost all in the first
  fully connected layer (7488 × 256).

### 5.4 The tricks that make it work, one by one

**1. Experience replay** (`ReplayBuffer`). Every step `(s, a, r, s', done, mask)` is
stored in a buffer of 100,000 steps (old ones get overwritten, it's a ring buffer).
Training uses **random batches of 64** from the buffer instead of the last step.
Why: consecutive steps are almost identical (strongly correlated), and gradient
descent assumes independent samples. Random sampling breaks the correlation, and
every experience is reused many times (more data efficient).

**2. Target network.** The target `r + γ max Q(s', a')` is computed with a **second
copy** of the network that is only updated every 1000 steps (copied from the main
one). Why: if the same network computes both the prediction and the target, every
update also moves the target, it's "chasing its own tail" and can oscillate or
diverge. Freezing the target for a while makes it a stable regression problem.

**3. Double DQN.** Normal DQN uses `max_a' Q_target(s', a')`. The max of noisy
estimates is biased upward (if all actions are really worth 0 but have noise, the
max is > 0): **overestimation bias**. Double DQN splits the job:

```
a* = argmax_a' Q_online(s', a')      # the online network picks the action
target = r + γ · Q_target(s', a*)    # the target network scores it
```

Because two different networks are used, their errors don't line up, and the bias
is much smaller (van Hasselt et al. 2016). Can be switched off with `--no-double`
(that's the ablation).

**4. Huber loss** (`F.smooth_l1_loss`). Squared error for small errors, absolute
error for big ones. Rare huge TD errors (e.g. the −500 for dying) would otherwise
produce huge gradients.

**5. Gradient clipping** at norm 10. Extra safety against exploding gradients.

**6. Reward scaling × 0.01.** Rewards go up to ±500; scaled they're ±5. Neural
networks train better when targets are around 1, not in the hundreds. The scale
doesn't change which policy is best, it just multiplies all Q-values by 0.01.

**7. Legal action masking.** When acting, only legal moves (not into a wall) are
considered. In the target, illegal next actions get −10^9 so `max` never picks them
(otherwise the target could use the Q-value of a move that isn't possible). The
buffer stores the legal mask of the next state for this.

**8. Training every 2 steps, after a warmup of 2000 steps.** Warmup: first fill the
buffer a bit so the first batches aren't all from the first game. Every 2 steps: a
speed / stability compromise.

### 5.5 Settings

| setting | value |
|---|---|
| optimizer | Adam, learning rate 2.5·10⁻⁴ |
| γ | 0.99 |
| batch size | 64 |
| buffer | 100,000 steps |
| target network update | every 1000 steps |
| warmup | 2000 steps |
| train every | 2 steps |
| ε | 1.0 → 0.05 linearly **per episode** over 60% of training (or a fixed number with `--eps-episodes`) |
| episodes | 3000 small, 5000 / 10000 / 6000 medium (different experiments) |

**Why γ = 0.99 for DQN but 0.9 for approximate Q-learning?** DQN has to figure out
from raw grids that walking toward food leads to +10 several steps later, and that
the +500 win is at the end; it needs a long horizon. Approximate Q-learning gets
`closest-food` as a feature, which already tells it the direction, so a short
horizon is enough and makes learning faster and more stable.

**Why ε per episode and not per step?** We first decayed it per step, but early
games are very short (Pac-Man dies after a few steps), so after 300 games ε was still
0.95. Per episode makes the schedule predictable.

### 5.6 The training loop (`run_episode` in `agents/training.py`)

```
obs = env.reset()
until the game ends:
    a = agent.act(env, obs)                # ε-greedy
    obs2, r, done, info = env.step(a)
    terminal = done and not truncated
    agent.remember(obs, a, r, obs2, terminal, legal mask of obs2)
        → store in buffer; every 2 steps sample a batch and do one gradient step;
          every 1000 steps copy the network into the target network
    obs = obs2
agent.end_episode()                         # ε goes down
```

### 5.7 Strengths and weaknesses

- \+ no hand made features; ends up better than approximate Q-learning (100% vs 96%
  small, ≈80% vs 63% medium) and scores higher because it eats more scared ghosts
- − needs about 10× more games (2,100 vs 230 on the small maze) and hours of CPU time
  (8-11 hours per run on the medium maze)
- − the network's input size depends on the maze, so a model only works on the maze
  it was trained on
- − on the medium maze it stops around 80%, while a planner shows 95% is possible

---

## 6. The agents that don't learn: baselines and the planner

These are **not** reinforcement learning. They are there to compare against.

### 6.1 Random agent (`agents/baselines.py`)
Picks any legal move. Wins 0%. Shows the game isn't trivial.

### 6.2 Food-only agent
Always walks to the closest food, ignores ghosts. Wins 61% on the small maze. Shows
that just running for the food isn't enough, the learned agents had to learn to
avoid ghosts (they win 98-100%).

### 6.3 The lookahead planner (`agents/planner.py`) – expectimax

It is given the **rules and the ghost behaviour**, which the RL agents never get.
It builds a search tree:

- **Max nodes**: Pac-Man's moves. Take the best one.
- **Chance nodes**: the ghosts' moves. Take the **average**, weighted by probability.
  A ghost with n allowed moves of which k are "chasing" moves picks each move with
  probability `0.8/n`, plus `0.2/k` extra for the chasing ones (exactly the rules in
  3.3). With 2 ghosts all combinations are multiplied out.
- Rewards along the way are the real game rewards; at the bottom of the tree (the
  depth limit) it adds `−(distance to the closest food)` so it keeps moving toward
  food.
- Ghosts too far away to reach Pac-Man within the lookahead are not branched on
  (they can't matter), which keeps it fast.

"Depth d" = Pac-Man looks d of his own moves ahead. Depth 5 takes 0.7 s per game.

Why we built it: every DQN version stopped at about 80% on the medium maze. Was 80%
the best possible (some games just can't be won), or was DQN making mistakes? The
planner answers it: looking 5 moves ahead wins 95%, so DQN still has room to improve.

Why **expectimax** and not **minimax**? Minimax assumes the ghosts play perfectly
against you. Ours are mostly random, so assuming the worst would make Pac-Man far too
scared. Expectimax uses the real probabilities.

---

## 7. Every file and every function

```
pacman/              the game
  layouts.py
  env.py
  ghosts.py
  observation.py
  render.py
  text_view.py
  live_view.py
agents/              the learning code
  features.py
  approx_q.py
  dqn.py
  replay_buffer.py
  baselines.py
  planner.py
  training.py
  loading.py
tests/
  test_env.py
  test_agents.py
  test_dqn.py
train.py, play.py, ghost_test.py, planner_test.py, generalization_test.py,
make_gif.py, run_experiments.py, summarize.py, plot_results.py
models/   results/
```

### pacman/layouts.py
- `LAYOUTS`: the three mazes as text.
- `read_layout(name)`: turns a maze into `walls` (a numpy bool grid), `food` (a set of
  (row, col)), `capsules`, `pacman` start, `ghosts` starts. `name` can be a maze name,
  the maze text itself (used by the tests) or a path to a text file with your own maze.
- `maze_distances(walls)`: runs BFS from every open cell, returns
  `dist[a][b]` = number of steps from a to b. Computed once, then every distance is a
  dictionary lookup.
- `bfs(walls, start)`: breadth-first search. Because every step costs 1, BFS gives
  shortest paths.

### pacman/env.py – `PacmanEnv`
A gym-style environment: `reset()` and `step(action)`.
- constants: `MOVES` (up, down, left, right as row/col changes), `ACTION_NAMES`, and
  the rewards.
- `__init__(layout, max_steps, ghost_chase_prob, scared_time, seed, ghost_directions)`:
  reads the maze, computes distances, makes its own random generator (`self.rng`,
  seeded, so ghost moves can be reproduced).
- `distance(a, b)`: maze distance (999 if unreachable).
- `next_position(pos, action)`: the cell after a move; a wall means stay.
- `legal_actions(pos)`: moves that don't hit a wall (for Pac-Man or a ghost).
- `legal_mask()`: the same as 4 True/False values, for DQN.
- `reset()`: everything back to the start, returns the first observation.
- `step(action)`: one turn in the order of section 3.4. Returns
  `(observation, reward, done, info)`; info has `won`, `truncated`, `score`.
- `check_ghosts()`: collisions: eat a scared ghost (+200, back home, `ghosts_eaten`
  +1) or die (−500).
- `use_ghost_directions(on)`: switches the 7th channel on/off.
- `get_observation()`: calls `make_observation`.

### pacman/ghosts.py
- `OPPOSITE`: the reverse of each move.
- `move_ghosts(env)`: moves every ghost by the rules of 3.3 using `env.rng`.
- `move_options(env, ghost, direction)`: legal moves without turning back.
- `chase_moves(env, ghost, options, scared, pacman)`: the moves that get closest to
  Pac-Man (or furthest if scared). Also used by the planner, so the planner's ghost
  model is exactly the real one.

### pacman/observation.py
- `make_observation(env)`: builds the channel grids of section 5.2.

### pacman/render.py
- `PygameView`: the game window. `draw` paints walls, food, capsules, ghosts (blue when
  scared), Pac-Man with a mouth pointing in his last direction, and a status line.
  `handle_events` notices when the window is closed (then training goes on without
  drawing instead of crashing) and passes key presses to `handle_key`: up arrow
  doubles the speed (up to 60 frames a second), down arrow halves it, space pauses.
  `wait_while_paused` shows "PAUSED" and waits (during `--live` training also stops
  while paused). `set_title`, `pause`, `close`.

### pacman/text_view.py
- `text_frame(env)`: the maze as text for the terminal (`play.py --text`).

### pacman/live_view.py
- `moving_average(values, window=50)`: average of the last 50 values, used for every
  curve in the project.
- `LiveTraining`: used by `train.py --live`. A matplotlib chart (score and win rate)
  that redraws every 10 episodes, and every N episodes the agent plays one game in the
  pygame window **without random moves** so you can watch it improve.

### agents/features.py
- `ghost_reach(env, ghost)`: the cells a ghost can be on after one move.
- `closest_distance(env, start, targets)`: maze distance to the nearest target.
- `extract_features(env, action)`: the 8 features of section 4.2, divided by 10.

### agents/approx_q.py – `ApproxQAgent`
- `weights`: a dict feature name → weight (missing = 0).
- `get_features(env, action)`: features, minus the ones dropped for an ablation.
- `q_value(features)`: the dot product w · f.
- `best_actions(env)`: the legal actions with the highest Q (all of them if tied) and
  that Q.
- `act(env, obs, greedy)`: ε-greedy; `greedy=True` for tests.
- `update(features, reward, env, terminal)`: the update of 4.3. `features` are from
  before the step, `env` is already in the next state.
- `end_episode()`: ε ← max(0.05, ε · decay).
- `save` / `load`: the weights as JSON.

### agents/dqn.py
- `QNetwork`: the CNN of 5.3; `forward` runs it.

### agents/replay_buffer.py
- `ReplayBuffer`: numpy arrays for obs, actions, rewards, next obs, done flags and
  next-state masks; `add` writes at position `pos` and wraps around (the oldest
  transition gets overwritten when it's full); `sample` picks random indices.

### agents/dqn.py (continued)
- `DQNAgent`:
  - `epsilon` (a property): linear from 1.0 to `eps_end` over `eps_decay_episodes`.
  - `act`: ε-greedy among legal moves.
  - `remember`: stores the scaled transition, trains every 2 steps after warmup,
    syncs the target network every 1000 steps.
  - `train_step`: samples 64 transitions, computes Q(s, a) for the actions taken,
    computes the (double) DQN target with masked next actions, Huber loss, backprop,
    gradient clipping, Adam step.
  - `end_episode`: counts episodes (for ε).
  - `save` / `load`: the network weights (`.pt`).

### agents/baselines.py
- `RandomAgent`, `FoodOnlyAgent` (section 6).

### agents/planner.py
- `State`: the parts of the game that change while planning; `copy()`.
- `PlannerAgent(depth)`:
  - `act`: tries each legal move and picks the best expected value.
  - `best_value`: max over Pac-Man's moves (max node).
  - `expected_value`: average over the outcomes of a move (chance node).
  - `evaluate`: the score at the bottom of the tree (−distance to closest food).
  - `outcomes`: simulates one step with the same event order as `PacmanEnv.step` and
    returns all `(probability, reward, next state)`.
  - `check_ghosts`: collisions in the simulated state.
  - `ghost_outcomes`: all combinations of ghost moves with their probabilities.
  - `ghost_moves`: one ghost's moves with probabilities.

### agents/training.py
- `run_episode(env, agent, kind)`: one training game (5.6); approximate Q-learning
  updates right after each step, DQN stores in the buffer.
- `evaluate(env, agent, games=100)`: plays games with `greedy=True`, returns win rate
  and average score. This is "the test".
- `keep_if_best(env, agent, model_file, best)`: plays 50 test games and saves the model
  if it's the best so far (used with `--eval-every`).

### agents/loading.py
- `model_extension(kind)`: `.json` or `.pt`.
- `model_path(kind, layout)`: the default model file, e.g. `models/dqn_small.pt`.
- `new_agent(...)`: creates a fresh agent with the ε schedule.
- `load_agent(kind, env, path)`: loads a saved model. For DQN it checks how many input
  channels the saved network has (6 or 7) and switches the ghost direction channel on
  if needed, so old and new models both work.

### Scripts
- `train.py`: trains one agent. Options: `--agent`, `--layout`, `--episodes`,
  `--ghost-chase`, `--max-steps`, `--seed`, `--name`, `--no-double`,
  `--drop-features`, `--eps-episodes`, `--eps-end`, `--eval-every`,
  `--ghost-directions`, `--threads`, `--live`, `--watch-every`, `--fps`. Prints progress every 50
  games, saves the model, the per-game log (`results/<name>.csv`: episode, score, won)
  and the final 100-game test (`results/<name>_test.json`).
- `play.py`: watch a trained model play (pygame window, or `--text` for the terminal).
- `ghost_test.py`: the ghost avoidance test (section 9.3).
- `planner_test.py`: plays games with the planner and saves the result.
- `generalization_test.py`: approximate Q-learning weights on every maze (9.6).
- `make_gif.py`: records a trained agent playing a won game as an animated gif
  (pygame draws into memory instead of a window, Pillow saves the frames). The gifs
  are in `results/`, a backup if the live demo doesn't work.
- `run_experiments.py`: every experiment × seeds 0, 1, 2, several at once (each one a
  separate process with 1 CPU thread). Skips runs that already finished, so it can be
  restarted.
- `summarize.py`: reads all experiment results, prints the mean ± std table and draws
  the charts (line = mean over seeds, band = ± 1 std).
- `plot_results.py`: learning curves for single runs.

### tests/
27 unit tests with `unittest` (`python -m unittest -v`).
- `test_env.py`: the game, on tiny corridor mazes where the outcome is certain:
  walking into a wall, eating food, winning, being caught, the capsule, eating a
  scared ghost (and the `ghosts_eaten` counter), the time limit, reset, the
  observation shape and the ghost direction channel.
- `test_agents.py`: the features, that a Q-learning update moves the weights the
  right way, dropped features, saving / loading, the food-only agent and the planner
  (goes for food, runs from a ghost).
- `test_dqn.py`: the network gives one Q-value per action, the replay buffer
  overwrites the oldest transitions, the agent only picks legal moves (random or
  greedy), ε goes down per episode, a training step really changes the network, and
  saving / loading gives the same network.

---

## 8. How the experiments were run and measured

- **Seeds.** Every experiment was trained 3 times with seeds 0, 1, 2. The seed fixes
  Python's, numpy's and torch's random numbers and the env's ghost generator. RL
  results vary a lot between runs, one run can be lucky; with 3 we can report a mean
  and a spread.
- **Mean ± std.** Over the 3 seeds, sample standard deviation.
- **Test.** After training, 100 games with no random moves (ε = 0).
- **Training curve.** The win rate over the last 50 training games (moving average).
  It's lower than the test because training still has random moves.
- **Episodes to 80% wins.** The first episode (after at least 50) where the training
  curve reaches 80%. Measures learning speed. "never" = didn't reach it.
- **Ablation.** Train again with one part removed, to see how much that part matters.
- **Precision of a 100-game test.** A win rate is a proportion, its standard error is
  `sqrt(p(1−p)/n)`. For p = 0.8, n = 100: `sqrt(0.16/100) = 0.04` → ±4%. So two
  results 3% apart are **not** really different. Important for interpreting 9.4.

---

## 9. All the results and what we learned from them

### 9.1 Main results (3 seeds each)

| experiment | test win rate | test avg score | episodes to 80% |
|---|---|---|---|
| approxq, small | 96% ± 1 | 883 ± 13 | 228 ± 27 |
| dqn, small | 100% ± 1 | 1073 ± 12 | 2138 ± 161 |
| approxq, medium | 63% ± 3 | 794 ± 56 | never |
| dqn, medium (best version) | 82% ± 6 | 1237 ± 68 | never in training |

Takeaways: both solve the small maze with every seed; approximate Q-learning learns
≈10× faster, DQN ends up a bit better and scores more (eats more scared ghosts). The
medium maze is much harder; DQN beats approximate Q-learning by about 20 points.

### 9.2 Ablations (small maze)

| experiment | test win rate | episodes to 80% |
|---|---|---|
| approxq, all features | 96% ± 1 | 228 |
| approxq **without ghost features** | **43% ± 1** | never |
| approxq without scared-ghost features | 97% ± 3 | 213 |
| dqn (double) | 100% ± 1 | 2138 ± 161 |
| dqn without double DQN | 98% ± 2 | 2423 ± 380 |

- The ghost features are essential: without them it can't see danger.
- The scared-ghost features don't matter on the small maze (the scared ghost is
  usually close anyway).
- Double DQN learned ≈300 episodes faster and was steadier, but final results are
  about the same. With 3 seeds that's not conclusive, and we say so.

### 9.3 Do they really avoid ghosts? (`ghost_test.py`, small maze, 200 games)

A **close call** = an active ghost comes within 2 steps; **escaped** = still alive 3
steps later. With ghosts chasing 20 / 50 / 80% of the time:
- food-only gets caught in 36-39% of games and escapes only 20-48% of close calls
- approximate Q-learning wins 98-100% and escapes 98-100%
- DQN wins 100% and escapes 100%, even against 80% chasers, which it never trained
  against
- different styles: approximate Q-learning keeps away (≈1 close call per game), DQN
  lets ghosts come close (5-7 per game) and always gets away, and eats more scared
  ghosts (1.7-1.9 per game against 0.9-1.2).
- The test uses seed 0 and gives every agent the same random numbers, so it can be
  repeated exactly.

A bug we found in the final code review: `ghost_test.py` used to count "a ghost was
eaten" whenever a step's reward was 199 or more, but the winning step gives 509, so
every won game counted one extra ghost. Now the game itself counts eaten ghosts
(`env.ghosts_eaten`) and the numbers above are from the fixed version. Good to
mention if asked about testing: reward-based detection was ambiguous, counting the
event where it happens is exact.

### 9.4 The medium maze story: how we tried to improve DQN

This is a good part to tell, because it shows the scientific process, including
the things that didn't work.

| step | change | test win rate |
|---|---|---|
| 1 | DQN, 5000 episodes | 73% ± 5 |
| 2 | 10000 episodes (curve was still going up) | 79% ± 4 |
| 3 | ε only over the first 2000 episodes + ghost direction channel, 6000 episodes | **82% ± 6** |
| 4 | step 3 + ε down to 0.01 + keep the best model (check every 250 episodes) | 78% ± 4 |

- **Step 2:** more episodes helped a bit, but the curve flattened after ≈8,500.
- **Diagnosis:** we played 200 games with each model and looked at how it lost:
  **every loss was a ghost catching it, late in the game** (≈12 of 50 food left),
  never the time limit. So the problem was ghosts, not speed.
- **Step 3:** (a) with ε decaying over 60% of 10,000 episodes, the agent made 40%
  random moves for thousands of games and died early, so it rarely practised the
  endgame. Shorter exploration: it started winning around episode 2,000 instead of
  6,000. (b) The network couldn't see which way ghosts were moving, which matters
  because ghosts never turn back → 7th channel. Result: same or slightly better
  (82%) with 40% fewer episodes. Both changes were made together, so we can't say
  which one helped more.
- **Step 4 didn't help.** The training curve went up to 70-85%, but only because
  there were fewer random moves during training, not because the agent got better.
  And "keep the best model" picked lucky checks: the best 50-game checks were 88-90%
  but the same models scored 74-81% in the final test. That's the **winner's curse**
  / selection bias: the maximum of several noisy measurements is biased upward.
- **Conclusion:** 78-82% are all within the ±4% measurement error, so DQN's
  performance plateaued around 80%.

### 9.5 The planner: how good can it get?

| agent (medium) | win rate |
|---|---|
| approxq | 63% |
| dqn | 82% |
| planner 1 move ahead | 75% |
| planner 2 | 87% |
| planner 3 | 89% |
| planner 4 | 91% |
| planner 5 | 95% |
| planner 6 | 95% |

So the game allows about 95%. Going from 5 to 6 moves ahead adds almost nothing
(94.8% → 95.4%), so 95% is close to the ceiling: about 5% of the games are lost
whatever you do (two ghosts happen to close in from both sides). DQN plays like a 1-2 move lookahead: it reacts well to
a ghost that's close, but doesn't plan far enough to avoid being trapped between two
ghosts. The natural next step would be **n-step returns** (learn from rewards several
steps ahead directly) or a **dueling network**.

### 9.6 Generalization: a maze it never saw (`generalization_test.py`)

We took the approximate Q-learning weights trained on the small maze and on the
medium maze and played 500 games on every maze, including `unseen` (never trained on).

| weights trained on | tiny | small | medium | unseen |
|---|---|---|---|---|
| small | 100% | 96% | 63% | 74% |
| medium | 100% | 97% | 59% | 70% |
| trained on that maze itself | | 96% | 63% | 71% ± 4 |

- The weights transfer **with no loss**: small-maze weights win 63% on the medium
  maze, the same as weights trained there, and 74% on the unseen maze, the same as
  training on it directly (3 seeds: 70%, 75%, 67%).
- Why: the features describe Pac-Man's surroundings ("ghost one step away",
  "distance to food"), not the maze, so the learned rule works anywhere. The limit on
  bigger mazes comes from the features (one step ahead), not from the training maze.
- DQN can't be tested this way: its network's input and fully connected layer are
  sized for one maze, and it learned that specific maze.

This is a nice point for the presentation: hand-made features need human work, but
they generalize; DQN learns its own features, but only for what it saw.

### 9.7 Are our ghosts harder than the usual ones? (random ghosts)

Other approximate Q-learning projects on the Berkeley Pac-Man game report around 90%
wins on their medium maze. Their default ghosts move completely at random, ours chase
pacman 20% of the time. So we trained the same agent on the medium maze with
`--ghost-chase 0` (ghosts still never turn back, like the Berkeley ones), 3 seeds:

| medium maze, 1000 episodes | test win rate | test avg score |
|---|---|---|
| ghosts chase 20% of the time | 63% ± 3 | 794 ± 56 |
| random ghosts | **74% ± 5** | **1003 ± 65** |

- Chasing ghosts cost about 11% of the wins (seeds: 76%, 69%, 78% vs 63% ± 3), and
  the training curve (`results/ghost_behavior.png`) is higher the whole time.
- It is still below 90%, so the ghosts explain only part of the gap. The rest is the
  maze (ours is not the same maze) and the one-step-ahead features. The 90% figure is
  also one student project tested on 40 games, not an exact target.
- How to say it: "our numbers can't be compared one to one with Berkeley projects
  because the game is ours, but when we make our ghosts behave like theirs we get
  closer to their results."

---

## 10. Questions you may be asked (with answers)

**Q: Why is this reinforcement learning and not supervised learning?**
There are no labels, nobody tells the agent the right move. It only gets rewards,
often delayed (the +500 comes at the end), and has to find out which earlier moves
caused them (the credit assignment problem). Its own actions also decide what data it
sees.

**Q: Is the planner reinforcement learning?**
No. It's given the rules and doesn't learn anything. It's a baseline to measure how
well the game can be played. The two RL algorithms are approximate Q-learning and DQN.

**Q: Model-free or model-based?**
Both RL algorithms are model-free: they never learn the rules, they learn Q directly
from experience. The planner is the opposite: it uses a perfect model.

**Q: Why not tabular Q-learning?**
Too many states (2^50 food combinations alone on the medium maze) and no
generalization. See 2.5.

**Q: Is Q-learning on-policy or off-policy? What's the difference with SARSA?**
Off-policy: the target uses `max Q(s', a')`, the best action, not the one actually
taken next. SARSA uses `Q(s', a_next)`, the action really taken (including random
ones), so it learns the value of the exploring policy. SARSA tends to be more careful
near danger while exploring.

**Q: What is γ and why those values?**
The discount: how much future rewards count. 0.99 for DQN (long horizon needed from
raw pixels), 0.9 for approximate Q-learning (features already point to food). See 5.5.

**Q: Why do you divide the features by 10?**
The update is `α · diff · f`. With large f and large diff (rewards of 500), the
weights jump too much and can diverge. Smaller features = smaller steps.

**Q: How did you choose the features?**
From what matters in the game: danger (ghost 1 step away, ghost distance), progress
(eats food, distance to food), and the bonus (capsule, scared ghosts). The ablation
shows the ghost features are the important ones.

**Q: Why does `eats-food` turn off when a ghost is near?**
Otherwise the food weight (+643) could cancel the danger weight and the agent would
risk death for one food. It's the classic design from the Berkeley Pac-Man project.

**Q: Why is ε not 0 at the end of training?**
A little exploration keeps updating Q for moves it otherwise never tries. In the test
we use ε = 0. We tried 0.01 instead of 0.05 (9.4), it didn't change the real result.

**Q: What does the replay buffer solve?**
Consecutive steps are correlated, gradient descent wants independent samples; and
each experience can be reused many times. See 5.4.

**Q: What does the target network solve?**
A moving target: without it, every update changes the target too and training can
oscillate or diverge. See 5.4.

**Q: What is overestimation bias / why double DQN?**
`max` over noisy estimates is biased upward; double DQN picks the action with one
network and evaluates it with the other. In our ablation it learned ≈300 episodes
faster, final results about the same.

**Q: Why Huber loss instead of MSE?**
Rare big errors (from ±500 rewards) would give huge gradients with MSE. Huber is
linear for big errors.

**Q: Why a CNN and not a fully connected network?**
The input is a grid; convolutions find local patterns (ghost next to Pac-Man) at any
position with shared weights, which needs much less data than learning every position
separately.

**Q: Why scale rewards by 0.01?**
Networks train best with targets around 1. Scaling all rewards by the same positive
number doesn't change the best policy.

**Q: Why does DQN need 10× more games?**
It starts from raw grids and must learn what a ghost, a corridor and food are. We gave
approximate Q-learning that knowledge in the features.

**Q: Why is the test win rate higher than the training curve?**
Training has random moves (ε), and one random move next to a ghost can kill you. The
test plays without random moves.

**Q: How do you know a result isn't luck?**
3 seeds with mean ± std, and we know a 100-game test is ±4%. We only call a difference
real when it's bigger than that (e.g. ghost features ablation: 96% vs 43%). We
explicitly say when it isn't (double DQN, steps 2-4 in 9.4).

**Q: Why does DQN stop around 80% on the medium maze?**
Every loss is a ghost catching it late in the game. The planner shows 95% is possible
with 5 moves of lookahead, and DQN plays like 1-2 moves. Learning long-term
consequences ("this corridor is a trap in 4 moves") from a delayed reward is hard;
n-step returns or longer training with a bigger buffer could help.

**Q: Why did "keep the best model" not help?**
Selection bias: we picked the highest of many noisy 50-game checks, so the best one
was partly luck; the fair 100-game test of the same models was lower. A fix would be
more games per check (e.g. 200), or averaging.

**Q: What's truncation and why does it matter?**
The game is cut off at 300 steps. That's not a real end, so we still bootstrap from
the next state instead of using value 0. See 3.5.

**Q: What is the action masking for?**
To never choose a move into a wall, and to never let the target use the value of an
impossible move.

**Q: Could the agent work on a new maze?**
Approximate Q-learning: yes, we tested it (9.6). Weights from the small maze win 74%
on a maze it never saw, the same as training on that maze directly. DQN: no, the
network's input size and fully connected layer depend on the maze size, and it
learned this specific maze. Making DQN generalize would need training on many
different mazes (and a network that doesn't depend on the size, e.g. only
convolutions plus global pooling).

**Q: Why does approximate Q-learning stop at 63% on the medium maze?**
Its features only look one step ahead at the closest ghost and the closest food. With
two ghosts it can walk into a corridor where both close in. The model is linear, so it
can't represent "this combination is dangerous" unless a feature describes it.
Our ghosts are also harder than the usual random ones: with random ghosts the same
agent wins 74% (see 9.7).

**Q: What would you do with more time?**
n-step returns and a dueling network for DQN; more seeds and 500-game tests to
measure small differences; a feature for "number of escape routes" for approximate
Q-learning; test generalization to other mazes.

**Q: What's the difference between expectimax and minimax?**
Minimax assumes the opponent plays the worst move for you; expectimax averages over
the opponent's moves with their probabilities. Our ghosts are mostly random, so
expectimax is the right model.

**Q: How long does training take?**
Approximate Q-learning: seconds. DQN small: ≈2 hours for 3000 episodes. DQN medium:
8-11 hours per run (CPU, 3 runs in parallel).

**Q: How much is the score in a won game?**
Food × 10 + 500 for the win − 1 per step + 20 per capsule + 200 per eaten ghost.
Around 900-1100 on the small maze, 1100-1500 on the medium.

**Q: What does the agent do when two moves are equally good?**
Approximate Q-learning picks randomly among them. DQN takes the first (in practice
the Q-values are almost never exactly equal).

---

## 11. Honest limitations

- DQN is trained and tested on the **same** maze (only the ghost randomness differs)
  and can't be used on another maze. Approximate Q-learning does generalize (9.6).
- 3 seeds is the minimum to see the spread; some small differences (double DQN,
  the DQN medium versions) are not conclusive.
- In step 3 of 9.4 two changes were made at once, so we can't separate their effects.
- Hyperparameters (learning rates, network size, γ) were chosen once, not tuned with
  a search.
- The replay buffer stores every observation twice (s and s') as float32: about
  650 MB per run on the medium maze. Fine here, but wasteful.
- Ghosts are simple (mostly random); real Pac-Man ghosts have smarter strategies.

---

## 12. How to give a live demo

```
python play.py --agent dqn                       # small maze, pygame window
python play.py --agent dqn --layout medium       # the best medium maze model (≈85%)
python play.py --agent approxq --layout unseen --model models/approxq_small.json
                                                 # small-maze weights on a new maze
python train.py --agent approxq --episodes 300 --live   # watch it learn in ~1 minute
python train.py --agent approxq --layout medium --live --fps 5   # medium maze, slow
python ghost_test.py --games 100
python planner_test.py --layout medium --depth 3 --games 50
python -m unittest -v
```

The `--live` demo is the best one for a lecture: the win rate chart climbs to around
90% within a few minutes (it reached 80% after about 230 games in our runs), and
every 15 episodes a game is shown so the audience can see
Pac-Man go from wandering into ghosts to clearly avoiding them. In the game window the
down arrow slows it down, the up arrow speeds it up and space pauses (handy when
explaining a move). Closing a window doesn't stop training, ctrl+c does.

If the laptop can't run Python, show the gifs in `results/` (`dqn_medium.gif`,
`dqn_small.gif`, `approxq_small.gif`, `approxq_medium.gif`).

Good to show as well: the learned weights table (4.5), the ablation chart
(`results/ablation_features.png`) and the medium maze chart with all DQN versions
(`results/medium_maze.png`).
