# Eliminates the phantom entropy values in the cached data.
import os

import pandas as pd
import yaml
from wandb.wandb_torch import torch

# Hard constants:
PROVIDER_UNAWARE_CACHE = "cache/"
PROVIDER_AWARE_CACHE = "cache4/"

# Steps:
if __name__ == "__main__":
    config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "config.yaml")
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    path_prefix = ""
    for prefix in config["path_prefix"]:
        if os.path.exists(prefix):
            path_prefix = prefix
            break

    if path_prefix == "":
        raise RuntimeError("No valid drive mounted.")

    # Get the list of providers
    data_path = os.path.normpath(
        os.path.join(path_prefix, config["audit_log_path"])
    )
    for provider in os.listdir(data_path):
        # Load the sequence indices for the provider aware and unaware models.
        provider_unaware_seq_indices = torch.load(
            os.path.normpath(os.path.join(path_prefix, provider, PROVIDER_UNAWARE_CACHE))
        )
        provider_aware_seq_indices = torch.load(
            os.path.normpath(os.path.join(path_prefix, provider, PROVIDER_AWARE_CACHE))
        )

        # Find all .csvs that begin with entropy-
        for cached_entropy in os.listdir(os.path.normpath(os.path.join(data_path, provider))):
            if not cached_entropy.startswith("entropy-") or not cached_entropy.endswith(".csv"):
                continue

            # Load the data from the cache.
            df = pd.read_csv(os.path.normpath(os.path.join(data_path, provider, cached_entropy)))

            # Rename the old cache file to something else.

            # 3. Iterate through each of the cached seqs, and eliminate the phantom entropy values beyond the given range.


