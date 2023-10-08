# Collects all cached versions of the entropy data and runs experiments atop them.
import os
import pandas as pd
import numpy as np
import torch
import yaml
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

import argparse

from model.modules import EHRAuditDataModule
from model.vocab import EHRVocab

# Things this needs to do:
# - Support both provider-aware and provider-unaware cache data.
# - Support multiple cached results, distinguishing between both sets.


if __name__ == "__main__":
    # Load the config.
    # Get the list of models from the config file
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

    # Get the list of saved models in our pretrained model path.
    model_paths = os.path.normpath(
        os.path.join(path_prefix, config["pretrained_model_path"])
    )
    # Get recursive list of subdirectories
    model_list = []
    for root, dirs, files in os.walk(model_paths):
        # If there's a .bin file, it's a model
        if any([file.endswith(".bin") for file in files]):
            # Append the last three directories to the model list
            model_list.append(os.path.join(*root.split(os.sep)[-3:]))

    if len(model_list) == 0:
        raise ValueError(f"No models found in {format(model_paths)}")

    # Load the data module and pull out the providers from there for the provider unaware models.from
    # All providers will be used (except for the one without one from the exclusion list).
    vocab = EHRVocab(
        vocab_path=os.path.normpath(os.path.join(path_prefix, config["vocab_path"]))
    )
    dm = EHRAuditDataModule(
        yaml_config_path=config_path,
        vocab=vocab,
        batch_size=1,  # Just one sample at a time
    )
    provider_unaware = [ds.provider for ds in dm.val_dataset.datasets] + [ds.provider for ds in dm.test_dataset.datasets]


