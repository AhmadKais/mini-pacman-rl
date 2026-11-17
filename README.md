# Mini Pac-Man with Reinforcement Learning

A small Pac-Man game where an agent learns to win on its own, using two methods:

1. **Approximate Q-learning**: a linear model over features I designed by hand
2. **DQN (Deep Q-Network)**: a CNN that learns straight from the grid

## The game

- Pac-Man wins by eating all the food (`.`)
- A ghost touching Pac-Man means game over
- Eating a capsule (`o`) scares the ghosts for 20 steps, and Pac-Man can eat them
- Ghosts mostly move randomly but sometimes chase (`--ghost-chase`, default 0.2)
- Mazes: `tiny`, `small` (1 ghost), `medium` (2 ghosts), see `pacman/layouts.py`
- A game ends after 300 steps at most

| event          | reward |
|----------------|--------|
| each step      | -1     |
| food           | +10    |
| capsule        | +20    |
| eat scared ghost | +200 |
| win            | +500   |
| caught         | -500   |

## Project structure

```
pacman/              the game
  layouts.py         maze maps, parser and maze distances (BFS)
  env.py             game rules (reset / step, like a gym env)
  ghosts.py          how the ghosts move
  observation.py     the grids the DQN sees
  render.py          terminal and pygame drawing
  live_view.py       live chart + game window while training
agents/              the learning code
  features.py        features for approximate Q-learning
  approx_q.py        approximate Q-learning agent
  dqn.py             DQN agent (CNN, replay buffer, target net, double DQN)
  baselines.py       random and food-only agents to compare against
  training.py        one training episode, and the test without random moves
  loading.py         creating / loading agents
tests/               unit tests for the game rules, features and agents
train.py             train one agent, save the model and a csv log
play.py              watch a trained agent play
ghost_test.py        check whether the agents really avoid ghosts
run_experiments.py   run every experiment with 3 seeds
summarize.py         results table and charts for the experiments
plot_results.py      learning curves for single runs
models/              trained models
results/             training logs and charts
```

## How to run

```
pip install -r requirements.txt

python train.py --agent approxq --episodes 500
python train.py --agent dqn --episodes 3000
python train.py --agent dqn --live          # watch it learn

python play.py --agent approxq
python play.py --agent dqn --text

python ghost_test.py --games 200
python -m unittest -v
```

To repeat all the experiments below (this takes several hours, mostly the DQN runs):

```
python run_experiments.py
python summarize.py
```

## How the algorithms work

**Approximate Q-learning.** Instead of a table with one entry per state,
Q(s,a) = Σ wᵢ·fᵢ(s,a). The features are things like "how many ghosts are one step
away", "distance to the closest food" and "does this move eat food". After every step:

```
diff = (r + γ · max_a' Q(s',a')) − Q(s,a)
wᵢ  ← wᵢ + α · diff · fᵢ(s,a)
```

It learns very fast (a few hundred games), and you can read the weights to see what
it learned, for example a big negative weight on `ghosts-1-step-away`.

**DQN.** The board is turned into 6 grids (walls, food, capsules, pacman, ghosts,
scared ghosts) and passed through a CNN that outputs one Q-value per action.
With `--ghost-directions` there is a 7th grid showing where each ghost just came from.
Some tricks keep training stable:
- a replay buffer, so it trains on random past transitions
- a target network that is updated every 1000 steps
- double DQN to reduce over-estimation
- rewards scaled by 0.01, Huber loss, gradient clipping
- epsilon-greedy exploration going from 1.0 down to 0.05

DQN needs far more games than approximate Q-learning, but it doesn't need any
hand made features.

## Results

Every experiment was trained 3 times with different seeds (0, 1, 2). The numbers are
mean ± standard deviation over the 3 runs. "Test" means 100 games without random
moves after training. "Episodes to 80% wins" is when the win rate over the last 50
training games first reaches 80%.

| experiment | test win rate | test avg score | episodes to 80% wins |
|---|---|---|---|
| approxq, small maze | 96% ± 1 | 883 ± 13 | 228 ± 27 |
| dqn, small maze | **100% ± 1** | **1073 ± 12** | 2138 ± 161 |
| approxq, medium maze | 63% ± 3 | 794 ± 56 | never |
| dqn, medium maze, 5000 episodes | 73% ± 5 | 1043 ± 36 | never (training reached ~40%) |
| dqn, medium maze, 10000 episodes | 79% ± 4 | 1170 ± 86 | never (training reached ~50%) |
| dqn, medium maze, improved, 6000 episodes | **82% ± 6** | **1237 ± 68** | never (training reached ~55%) |

![small maze](results/small_maze.png)
![medium maze](results/medium_maze.png)

- Both methods solve the small maze with every seed, so the result is not luck.
- Approximate Q-learning learns about **10 times faster** (around 230 episodes, under
  a minute) but DQN ends up a bit better, mostly because it eats more scared ghosts.
- On the medium maze (2 ghosts, about 4 times the food) approximate Q-learning only
  wins 63%. Probably because its features only look at the closest ghost and the
  closest food, which is not enough to plan around two ghosts in a bigger maze.
- DQN does better on the medium maze but needs a lot more time. With 5000 episodes
  it wins 73%, and it wins almost nothing until the random moves drop low (around
  episode 2,700). Each run took about 8 hours.
- Its curve was still going up at episode 5000, so I trained it again with 10000
  episodes (about 11.5 hours per run). That got it to **79% wins** and a higher score.
  The training curve flattens out around 45-50% after episode 8,500, so doubling
  the episodes again would probably give only a small gain.
- Almost every game the 10000 episode agent lost was a ghost catching it late in the
  game (about 12 of the 50 food left), not running out of time. So the **improved**
  run changes two things: the random moves fade out over the first 2000 episodes
  instead of 60% of the training (`--eps-episodes 2000`), and the network gets a 7th
  grid with the cell each ghost just came from (`--ghost-directions`). Ghosts never
  turn back, so this shows which way they are going.
- The improved agent starts winning around episode 2,000 instead of 6,000 and gets
  **82% wins with 6000 episodes**, slightly better than 79% with 10000. The gain is
  within the spread between seeds (75%, 84% and 87%), so the clear win is that it
  gets there with 40% fewer episodes. Both changes were made together, so this
  doesn't say which one helped more.
- The test win rate is higher than the training curve because the test has no
  random moves.

### Ablations

What happens when parts of the method are taken away (small maze, 3 seeds each):

| experiment | test win rate | test avg score | episodes to 80% wins |
|---|---|---|---|
| approxq (all features) | 96% ± 1 | 883 ± 13 | 228 ± 27 |
| approxq without ghost features | **43% ± 1** | **215 ± 39** | never |
| approxq without scared-ghost features | 97% ± 3 | 876 ± 16 | 213 ± 27 |
| dqn (double DQN) | 100% ± 1 | 1073 ± 12 | 2138 ± 161 |
| dqn without double DQN | 98% ± 2 | 1083 ± 12 | 2423 ± 380 |

![feature ablation](results/ablation_features.png)
![double dqn ablation](results/ablation_double_dqn.png)

- **The ghost features are what makes approximate Q-learning work.** Without
  `ghosts-1-step-away` and `ghost-closeness` the agent can't see danger, and the win
  rate stays around 30% during training (43% in the test).
- The scared-ghost features (`eats-scared-ghost`, `scared-ghost-closeness`) don't change
  the result. On this small maze the scared ghost is usually close anyway, so hunting
  it on purpose doesn't add much.
- **Double DQN helps a little.** It reaches 80% wins about 300 episodes earlier and
  its training win rate stays around 80% instead of 65-75%, but the final test results
  are almost the same. With only 3 seeds and overlapping error bars, this difference
  is small and not conclusive.

### Do they really avoid the ghosts?

`ghost_test.py` plays 200 games per agent and compares them with two baselines:
**random** (any legal move) and **food-only** (always walks to the closest food and
ignores the ghosts). A *close call* is an active ghost coming within 2 steps of
pacman; it counts as *escaped* if pacman is still alive 3 steps later.
The agents were trained with ghosts chasing 20% of the time, and tested against
more aggressive ghosts as well.

| ghosts chase | agent | win | caught | close calls / game | escaped | ghosts eaten / game |
|---|---|---|---|---|---|---|
| 20% | random | 0% | 100% | 1.5 | 41% | 0.21 |
| 20% | food-only | 61% | 39% | 0.7 | 44% | 1.50 |
| 20% | approxq | 98% | 2% | 0.8 | 99% | 2.15 |
| 20% | dqn | **100%** | **0%** | 4.8 | **100%** | **2.84** |
| 50% | random | 0% | 100% | 1.4 | 35% | 0.14 |
| 50% | food-only | 64% | 36% | 0.5 | 35% | 1.35 |
| 50% | approxq | 97% | 3% | 1.0 | 97% | 2.00 |
| 50% | dqn | **100%** | **0%** | 6.3 | **100%** | **2.77** |
| 80% | random | 0% | 100% | 1.2 | 19% | 0.06 |
| 80% | food-only | 76% | 24% | 0.3 | 18% | 1.20 |
| 80% | approxq | 100% | 0% | 1.0 | 99% | 1.79 |
| 80% | dqn | **100%** | **0%** | 7.0 | **100%** | **2.71** |

- Just running for the food isn't enough: food-only gets caught in a quarter to a
  third of its games, and escapes fewer close calls as the ghosts get more aggressive.
- Both trained agents escape almost every close call, even against ghosts that chase
  80% of the time, which they never trained against.
- They play differently. Approximate Q-learning stays away from ghosts (about 1 close
  call per game). DQN lets ghosts get close 5 to 7 times a game and always gets away,
  and it eats more scared ghosts, which is where its higher score comes from.

## Tests

`tests/` has 18 unit tests. The game rules are tested on tiny corridor mazes where the
outcome is known: walls, eating food, winning, getting caught, capsules, eating a scared
ghost, the time limit and reset. The agent tests check the features, the Q-learning
update, saving / loading and the food-only baseline.

```
python -m unittest -v
```
