# Loads all datasets for each physician and calculates the characteristics for each dataset.
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from tabulate import tabulate
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
import yaml

import sys

from model.data import EHRAuditDataset

if __name__ == "__main__":
    # Get the path of the data from config
    with open(
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "config.yaml")),
        "r",
    ) as f:
        config = yaml.safe_load(f)

    path_prefix = ""
    for prefix in config["path_prefix"]:
        if os.path.exists(prefix):
            path_prefix = prefix
            break

    metric_xlsx = pd.read_excel(
        os.path.normpath(os.path.join(path_prefix, config["metric_name_dict"]["file"])),
        engine="openpyxl",
    )

    metric_dict = dict({k: 0 for k in metric_xlsx["METRIC_NAME"]})
    missing_metric_dict = defaultdict(int)

    # Load the datasets
    providers = []
    for provider in os.listdir(
        os.path.normpath(os.path.join(path_prefix, config["audit_log_path"]))
    ):
        prov_path = os.path.normpath(
            os.path.join(path_prefix, config["audit_log_path"], provider)
        )
        # Check the file is not empty and exists, there's a couple of these.
        log_path = os.path.normpath(os.path.join(prov_path, config["audit_log_file"]))
        if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
            continue

        providers.append(prov_path)

    def find_missing_metrics(provider_path):
        dataset = EHRAuditDataset(
            provider_path,
            sep_min=config["sep_min"],
            log_name=config["audit_log_file"],
            should_tokenize=False,
            cache=None,
        )
        dataset.load_from_log()
        missing_ms = defaultdict(int)
        not_missing_ms = defaultdict(int)
        for seq in dataset:
            for event in seq["METRIC_NAME"]:
                if event not in metric_dict:
                    missing_ms[event] += 1
                else:
                    not_missing_ms[event] += 1

        return missing_ms, not_missing_ms

    # Iterate over the datasets and find the missing metrics in parallel
    missing_metrics = Parallel(n_jobs=-1, verbose=2)(
        delayed(find_missing_metrics)(provider_path) for provider_path in providers
    )

    # Join the dictionaries
    for mm, nmm in tqdm(missing_metrics, desc="Joining dictionaries"):
        for key, value in mm.items():
            missing_metric_dict[key] += value
        for key, value in nmm.items():
            metric_dict[key] += value

    # Save the results to an xlsx file
    missing_metric_df = pd.DataFrame.from_dict(missing_metric_dict, orient="index")
    missing_metric_df.to_excel(
        os.path.normpath(
            os.path.join(
                path_prefix, config["results_path"], "missing_metric_names.xlsx"
            )
        )
    )
    not_missing_metric_df = pd.DataFrame.from_dict(metric_dict, orient="index")
    not_missing_metric_df.to_excel(
        os.path.normpath(
            os.path.join(
                path_prefix, config["results_path"], "not_missing_metric_names.xlsx"
            )
        )
    )
