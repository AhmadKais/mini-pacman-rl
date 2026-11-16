# Summarizes the runs from run_experiments.py: a table with the mean and the
# spread over the seeds, and learning curve charts (line = mean, band = +- 1 std).
#   python summarize.py

import csv
import glob
import json
import os
import statistics

import matplotlib.pyplot as plt

from pacman.live_view import moving_average

FOLDER = "results/experiments"

CHARTS = {
    "small_maze.png": ["approxq_small", "dqn_small"],
    "medium_maze.png": ["approxq_medium", "dqn_medium", "dqn_medium_10k"],
    "ablation_double_dqn.png": ["dqn_small", "dqn_small_no_double"],
    "ablation_features.png": ["approxq_small", "approxq_small_no_ghost_features",
                              "approxq_small_no_scared_features"],
}


def experiment_name(path):
    # "results/experiments/dqn_small_seed2_test.json" -> "dqn_small"
    base = os.path.basename(path)
    return base[:base.rindex("_seed")]


def find_runs():
    # experiment name -> list of run paths without the extension
    runs = {}
    for path in sorted(glob.glob(FOLDER + "/*_test.json")):
        name = experiment_name(path)
        if name not in runs:
            runs[name] = []
        runs[name].append(path[:-len("_test.json")])
    return runs


def win_curve(run):
    wins = []
    with open(run + ".csv") as f:
        for row in csv.DictReader(f):
            wins.append(float(row["won"]) * 100)
    return moving_average(wins)


def episodes_to_80(curve):
    for i in range(len(curve)):
        if i >= 50 and curve[i] >= 80:
            return i + 1
    return None


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def print_table(runs):
    print("| experiment | seeds | test win rate | test avg score | episodes to 80% wins |")
    print("|---|---|---|---|---|")
    for name in runs:
        win_rates = []
        scores = []
        reached = []
        for run in runs[name]:
            with open(run + "_test.json") as f:
                result = json.load(f)
            win_rates.append(result["win_rate"] * 100)
            scores.append(result["avg_score"])
            episode = episodes_to_80(win_curve(run))
            if episode is not None:
                reached.append(episode)

        win_mean, win_std = mean_std(win_rates)
        score_mean, score_std = mean_std(scores)
        if len(reached) == 0:
            speed = "never"
        else:
            speed_mean, speed_std = mean_std(reached)
            speed = f"{speed_mean:.0f} ± {speed_std:.0f} ({len(reached)}/{len(runs[name])} runs)"
        print(f"| {name} | {len(runs[name])} | {win_mean:.0f}% ± {win_std:.0f} "
              f"| {score_mean:.0f} ± {score_std:.0f} | {speed} |")


def plot_chart(filename, names, runs):
    plt.figure(figsize=(7, 4.5))
    for name in names:
        if name not in runs:
            continue
        curves = []
        length = None
        for run in runs[name]:
            curve = win_curve(run)
            curves.append(curve)
            if length is None or len(curve) < length:
                length = len(curve)
        means = []
        lows = []
        highs = []
        for i in range(length):
            values = []
            for curve in curves:
                values.append(curve[i])
            mean, std = mean_std(values)
            means.append(mean)
            lows.append(mean - std)
            highs.append(mean + std)
        episodes = range(1, length + 1)
        plt.plot(episodes, means, label=f"{name} ({len(curves)} seeds)")
        plt.fill_between(episodes, lows, highs, alpha=0.2)

    plt.title("win rate % (moving avg of 50), mean ± std over seeds")
    plt.xlabel("episode")
    plt.ylim(-5, 105)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/" + filename, dpi=120)
    plt.close()
    print("saved results/" + filename)


def main():
    runs = find_runs()
    if len(runs) == 0:
        print("no finished runs yet, start them with: python run_experiments.py")
        return
    print_table(runs)
    print()
    for filename in CHARTS:
        plot_chart(filename, CHARTS[filename], runs)


if __name__ == "__main__":
    main()
