# Plot the learning curves from the csv files that train.py writes.
#   python plot_results.py results/approxq_small.csv results/dqn_small.csv

import csv
import os
import sys

import matplotlib.pyplot as plt

from pacman.live_view import moving_average


def read_log(path):
    episodes = []
    scores = []
    wins = []
    with open(path) as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            scores.append(float(row["score"]))
            wins.append(float(row["won"]) * 100)
    return episodes, scores, wins


def main():
    files = sys.argv[1:]
    if len(files) == 0:
        print("usage: python plot_results.py results/<file>.csv [more files]")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    score_ax = axes[0]
    win_ax = axes[1]
    for path in files:
        episodes, scores, wins = read_log(path)
        name = os.path.splitext(os.path.basename(path))[0]
        score_ax.plot(episodes, moving_average(scores), label=name)
        win_ax.plot(episodes, moving_average(wins), label=name)

    score_ax.set_title("Score (moving avg of 50 episodes)")
    win_ax.set_title("Win rate % (moving avg of 50 episodes)")
    for ax in [score_ax, win_ax]:
        ax.set_xlabel("episode")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/learning_curves.png", dpi=120)
    print("saved results/learning_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
