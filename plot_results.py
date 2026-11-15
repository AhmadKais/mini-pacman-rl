# Plot learning curves from the csv files that train.py writes.
#   python plot_results.py results/approxq_small.csv results/dqn_small.csv

import csv
import sys

import matplotlib.pyplot as plt


def moving_avg(values, k=50):
    out = []
    for i in range(len(values)):
        window = values[max(0, i - k + 1):i + 1]
        out.append(sum(window) / len(window))
    return out


def main():
    files = sys.argv[1:]
    if not files:
        print("usage: python plot_results.py results/<file>.csv [more files...]")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for path in files:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        eps = [int(r["episode"]) for r in rows]
        scores = [float(r["score"]) for r in rows]
        wins = [float(r["won"]) * 100 for r in rows]
        name = path.replace("\\", "/").split("/")[-1].replace(".csv", "")
        ax1.plot(eps, moving_avg(scores), label=name)
        ax2.plot(eps, moving_avg(wins), label=name)

    ax1.set_title("Score (moving avg of 50 episodes)")
    ax2.set_title("Win rate % (moving avg of 50 episodes)")
    for ax in (ax1, ax2):
        ax.set_xlabel("episode")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/learning_curves.png", dpi=120)
    print("saved results/learning_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
