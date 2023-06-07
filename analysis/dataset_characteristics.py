# Loads all datasets for each physician and calculates the characteristics for each dataset.
import argparse
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib
from tabulate import tabulate
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Tuple, List
import yaml

from model.data import EHRAuditDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Arguments:
    # - Single or all providers
    # - What delineates a shift (default > 4 hours)
    # - Which of the log files to use

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", type=str, default=None, help="The provider to analyze."
    )
    parser.add_argument(
        "--shift", type=int, default=4, help="The number of hours to delineate a shift."
    )
    parser.add_argument("--log", type=str, default="log", help="The log file to use.")

    args = parser.parse_args()

    # Get the path of the data from config
    with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    data_path = config["audit_data_path"]

    datasets = []
    if args.provider is None:
        for provider in os.listdir(data_path):
            datasets.append(EHRAuditDataset(provider, args.shift, args.log))
    else:
        prov_path = os.path.join(data_path, args.provider)
        datasets.append(EHRAuditDataset(prov_path, args.shift, args.log))

    # Calculate the characteristics for each dataset one at a time.
    statistics = defaultdict(list)
    for dataset in tqdm(datasets):
        # Key characteristics are:
        # - Number of patients
        # - Number of events in a shift.
        # - Distribution of time deltas.
        patients_per_shift = []
        events_per_shift = []
        mean_time_delta = []
        min_time_delta = []
        max_time_delta = []
        std_time_delta = []
