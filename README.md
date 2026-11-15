# Mini Pac-Man with Reinforcement Learning

A small Pac-Man game where an agent learns to win on its own, using two methods:

1. **Approximate Q-learning**: a linear model over features I designed by hand
2. **DQN (Deep Q-Network)**: a CNN that learns straight from the grid

## The game

- Pac-Man wins by eating all the food (`.`)
- A ghost touching Pac-Man means game over
- Eating a capsule (`o`) scares the ghosts for 20 steps, and Pac-Man can eat them
- Ghosts mostly move randomly but sometimes chase (`--ghost-chase`, default 0.2)
- Mazes: `tiny`, `small`, `medium` (see `layouts.py`)

| event          | reward |
|----------------|--------|
| each step      | -1     |
| food           | +10    |
| capsule        | +20    |
| eat scared ghost | +200 |
| win            | +500   |
| caught         | -500   |

## Files

| file | what it does |
|------|--------------|
| `layouts.py` | maze maps and the parser |
| `pacman_env.py` | game logic (reset / step, like a gym env) |
| `features.py` | features for approximate Q-learning |
| `approx_q_agent.py` | approximate Q-learning agent |
| `dqn_agent.py` | DQN agent (replay buffer, target net, double DQN) |
| `train.py` | training loop, saves the model and a csv log |
| `play.py` | watch a trained agent play (pygame or terminal) |
| `render.py` | drawing code |
| `plot_results.py` | learning curves |

## How to run

```
pip install -r requirements.txt

python train.py --agent approxq --episodes 500
python train.py --agent dqn --episodes 3000

python play.py --agent approxq
python play.py --agent dqn --text

python plot_results.py results/approxq_small.csv results/dqn_small.csv
```

Try a bigger maze with `--layout medium` (it needs more episodes for DQN).

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
Some tricks keep training stable:
- a replay buffer, so it trains on random past transitions
- a target network that is updated every 1000 steps
- double DQN to reduce over-estimation
- rewards scaled by 0.01, Huber loss, gradient clipping
- epsilon-greedy exploration going from 1.0 down to 0.05

DQN needs far more games than approximate Q-learning, but it doesn't need any
hand made features.
