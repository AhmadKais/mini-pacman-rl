# Runs every experiment with several seeds, a few trainings at the same time.
# Runs that already finished are skipped, so it can be stopped and started again.
#   python run_experiments.py
#   python run_experiments.py --only dqn_small dqn_small_no_double --workers 4
# Afterwards: python summarize.py

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

SEEDS = [0, 1, 2]

# the slowest ones come first so the workers stay busy until the end
EXPERIMENTS = {
    # shorter exploration (2000 episodes) and the ghost direction channel
    "dqn_medium_improved": ["--agent", "dqn", "--layout", "medium", "--episodes", "6000",
                            "--eps-episodes", "2000", "--ghost-directions"],
    "dqn_medium_10k": ["--agent", "dqn", "--layout", "medium", "--episodes", "10000"],
    "dqn_medium":["--agent", "dqn", "--layout", "medium", "--episodes", "5000"],
    "dqn_small": ["--agent", "dqn", "--layout", "small", "--episodes", "3000"],
    "dqn_small_no_double": ["--agent", "dqn", "--layout", "small", "--episodes", "3000",
                            "--no-double"],
    "approxq_medium": ["--agent", "approxq", "--layout", "medium", "--episodes", "1000"],
    "approxq_small": ["--agent", "approxq", "--layout", "small", "--episodes", "500"],
    "approxq_small_no_ghost_features": [
        "--agent", "approxq", "--layout", "small", "--episodes", "500",
        "--drop-features", "ghosts-1-step-away", "ghost-closeness"],
    "approxq_small_no_scared_features": [
        "--agent", "approxq", "--layout", "small", "--episodes", "500",
        "--drop-features", "eats-scared-ghost", "scared-ghost-closeness"],
}


def run_job(experiment, seed):
    name = f"experiments/{experiment}_seed{seed}"
    if os.path.exists("results/" + name + "_test.json"):
        print(f"skip  {name} (already done)", flush=True)
        return
    command = [sys.executable, "train.py"] + EXPERIMENTS[experiment]
    command += ["--seed", str(seed), "--name", name, "--threads", "1"]
    print(f"start {name}", flush=True)
    start = time.time()
    with open("results/" + name + ".log", "w") as log_file:
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
    minutes = (time.time() - start) / 60
    if result.returncode == 0:
        print(f"done  {name} in {minutes:.1f} min", flush=True)
    else:
        print(f"FAILED {name}, see results/{name}.log", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="experiment names to run")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    names = list(EXPERIMENTS)
    if args.only is not None:
        names = args.only
    os.makedirs("results/experiments", exist_ok=True)

    jobs = []
    for experiment in names:
        for seed in SEEDS:
            jobs.append((experiment, seed))
    print(f"{len(jobs)} runs, {args.workers} at a time")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for experiment, seed in jobs:
            pool.submit(run_job, experiment, seed)
    print("all done, now run: python summarize.py")


if __name__ == "__main__":
    main()
