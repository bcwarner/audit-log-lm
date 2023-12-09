# Joins a list of tensorboard runs into a single plot.
import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.01)
    args = p.parse_args()

    # Get all csv files in the directory
    files = [f for f in os.listdir(os.path.join(os.getcwd(), "results")) if ("train" in f)]
    # Split up files by model type
    files_split = defaultdict(list)
    for f in files:
        model_type = f.split("_")[2:5]
        model_type = f"{model_type[0]}-{model_type[1]}.{model_type[2]}"
        files_split[model_type].append(f)

    points_by_model = defaultdict(list)
    for model_type, fnames in files_split.items():
        # Load all files into dataframes
        dfs = []
        for fname in fnames:
            dfs.append(pd.read_csv(os.path.join(os.getcwd(), "results", fname)))
        df = pd.concat(dfs, ignore_index=True)
        df = df.groupby("Step").last().reset_index()
        points_by_model[model_type] = [df["Step"].values, np.exp(df["Value"].ewm(alpha=args.alpha).mean().values)]

    # Plot all models
    fig, ax = plt.figure(), plt.axes()
    fig.set_size_inches(7, 5)
    for model_type, points in points_by_model.items():
        ax.semilogy(*points_by_model[model_type], label=model_type)

    ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    plt.title("Smoothed Training Perplexity")
    plt.tight_layout()
    plt.savefig(os.path.join(os.curdir, "train_loss.pdf"))
