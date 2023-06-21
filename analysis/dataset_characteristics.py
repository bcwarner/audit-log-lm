# Loads all datasets for each physician and calculates the characteristics for each dataset.
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from tabulate import tabulate
from tqdm import tqdm
from collections import defaultdict
import yaml

import sys

sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))

from model.data import EHRAuditDataset

if __name__ == "__main__":
    # Arguments:
    # - Single or all providers
    # - What delineates a shift (default > 4 hours)
    # - Which of the log files to use

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", type=str, default=None, help="The provider to analyze."
    )

    args = parser.parse_args()

    # Get the path of the data from config
    with open(
        os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "config.yaml"), "r"
        )
    ) as f:
        config = yaml.safe_load(f)
    data_path = config["audit_log_path"]
    log_name = config["audit_log_file"]
    sep_min = config["sep_min"]
    results_path = config["results_path"]

    # Load the datasets
    datasets = []
    if args.provider is None:
        for provider in os.listdir(data_path):
            prov_path = os.path.normpath(os.path.join(data_path, provider))
            # Check the file is not empty and exists, there's a couple of these.
            log_path = os.path.normpath(os.path.join(prov_path, log_name))
            if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                continue
            datasets.append(
                EHRAuditDataset(prov_path, sep_min=sep_min, log_name=log_name)
            )
    else:
        prov_path = os.path.normpath(os.path.join(data_path, args.provider))
        datasets.append(EHRAuditDataset(prov_path, sep_min=sep_min, log_name=log_name))

    # Relevant columns
    patient_col = datasets[0].user_col
    event_cols = datasets[0].event_type_cols
    time_col = datasets[0].timestamp_col

    # Summarize each provider in parallel.
    def summarize(dataset):
        # Calculate the characteristics for each dataset one at a time.
        # Key characteristics are:
        # - Number of patients
        # - Number of events in a shift.
        # - Distribution of time deltas.

        print(f"Summarizing {dataset.provider}")
        statistics = defaultdict(list)
        for shift in dataset:
            # Calculate the number of sessions (i.e. gaps of 5 or more minutes)
            statistics["Sessions"].append(len(shift[shift[time_col] > 5 * 60]) + 1)
            statistics["Patients"].append(shift[patient_col].nunique())
            statistics["Events"].append(len(shift))
            statistics["Events/Session"].append(
                statistics["Events"][-1] / statistics["Sessions"][-1]
            )
            statistics["Mean Time Delta"].append(shift[time_col].mean())
            statistics["Min Time Delta"].append(shift[time_col].min())
            statistics["Max Time Delta"].append(shift[time_col].max())
            statistics["Std Time Delta"].append(shift[time_col].std())
        return statistics

    # Summarize each provider in parallel.
    from joblib import Parallel, delayed

    n_jobs = -1 if args.provider is None else 1
    par_statistics = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(summarize)(dataset) for dataset in datasets
    )

    # Combine the statistics from each provider.
    statistics = defaultdict(list)
    for provider in par_statistics:
        for k, v in provider.items():
            statistics[k].extend(v)

    summary_stats = defaultdict(float)
    # After iterating through all datasets, calculate the statistics of each dataset.
    for k, v in statistics.items():
        stats = np.array(v)
        summary_stats[f"{k} Provider Mean"] = stats.mean()
        summary_stats[f"{k} Provider Std"] = np.nanstd(stats)
        summary_stats[f"{k} Provider Min"] = np.min(stats)
        summary_stats[f"{k} Provider Max"] = np.max(stats)

    # Print the summary statistics.
    print(tabulate(summary_stats.items(), tablefmt="pretty"))

    # Save the LaTeX table and a tsv.
    with open(
        os.path.normpath(os.path.join(results_path, "dataset_characteristics.tex")), "w"
    ) as f:
        f.write(tabulate(summary_stats.items(), tablefmt="latex"))
    with open(
        os.path.normpath(os.path.join(results_path, "dataset_characteristics.tsv")), "w"
    ) as f:
        f.write(tabulate(summary_stats.items(), tablefmt="tsv"))

    def histogram(data, title, xlabel, ylabel, filename, semilog=False):
        plt.clf()
        if semilog:
            logbins = np.logspace(
                np.log10(np.min(data) + 1), np.log10(np.max(data)), 50
            )
            plt.hist(data, bins=logbins)
            plt.xscale("log")
        else:
            plt.hist(data, bins=100)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.show()
        plt.savefig(os.path.normpath(os.path.join(results_path, filename + ".png")))
        tikzplotlib.save(
            os.path.normpath(os.path.join(results_path, filename + ".tex"))
        )

    # Plot the distribution of time deltas.
    histogram(
        statistics["Mean Time Delta"],
        f"Mean Time Delta/Shift (n={len(statistics['Mean Time Delta'])})",
        "Mean Time Delta (s)",
        "Frequency",
        "mean_time_delta",
        semilog=True,
    )

    histogram(
        statistics["Patients"],
        f"Number of Patients/Shift (n={len(statistics['Patients'])})",
        "Number of Patients",
        "Frequency",
        "num_patients",
        semilog=True,
    )

    histogram(
        statistics["Events/Session"],
        f"Number of Events/Session (n={len(statistics['Events/Session'])})",
        "Number of Events/Session",
        "Frequency",
        "num_events_session",
        semilog=True,
    )

    histogram(
        statistics["Events"],
        f"Number of Events/Shift (n={len(statistics['Events'])})",
        "Number of Events/Shift",
        "Frequency",
        "num_events",
        semilog=True,
    )
