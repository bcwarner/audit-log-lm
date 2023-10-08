# Collects all cached versions of the entropy data and runs experiments atop them.
import os
import pickle

import pandas as pd
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

import argparse

from model.modules import EHRAuditDataModule
from model.vocab import EHRVocab

# Things this needs to do:
# - Support both provider-aware and provider-unaware cache data.
# - Support multiple cached results, distinguishing between both sets.

class Experiment:
    def __init__(
        self,
        config: dict,
        path_prefix: str,
        vocab: EHRVocab,
        model: str,
        *args,
        **kwargs,
    ):
        self.config = config
        self.path_prefix = path_prefix
        self.vocab = vocab
        self.model = model

    def requirements(self):
        """
        Returns a list of requirements for this experiment.
        """
        return {
            "logs": False,
            "provider_aware": False,
            "provider_unaware": False,
            "comparison": "union", # Can be union or intersection
        }

    def _exp_cache_path(self):
        return os.path.normpath(
            os.path.join(
                self.path_prefix,
                self.config["results_path"],
                f"exp_cache_{self.__class__.__name__}",
            )
        )

    def map(self,
            provider=None,
            audit_log_df: pd.DataFrame = None,
            provider_aware_df: Dict[str, pd.DataFrame] = None,
            provider_unaware_df: Dict[str, pd.DataFrame] = None,
            ):
        return None

    def on_finish(self, results: Dict[str, pd.DataFrame]):
        return None

    def plot(self):
        # Reset matplotlib figure size, etc.
        plt.rcParams.update(plt.rcParamsDefault)
        plt.clf()
        plt.gcf().set_size_inches(5, 5)

class PerFieldEntropyExperiment(Experiment):
    # Just records the entropy of each field as well as overall.
    def __init__(self, config, path_prefix, vocab, model, *args, **kwargs):
        super().__init__(config, path_prefix, vocab, model, *args, **kwargs)
        self.field_entropies = defaultdict(list)
        self.row_entropies = []
        self._samples_seen = 0

    def requirements(self):
        return {
            "logs": False,
            "provider_aware": True,
            "provider_unaware": True,
            "comparison": "union",
        }

    def map(self,
            provider=None,
            audit_log_df: pd.DataFrame = None,
            provider_aware_df: Dict[str, pd.DataFrame] = None,
            provider_unaware_df: Dict[str, pd.DataFrame] = None,
            ):
        # field => model => entropy count, average, std
        results = defaultdict(lambda: defaultdict(int))
        for k, df in {**provider_aware_df, **provider_unaware_df}.items():
            # Iterate each of the fields in the df and aggregate the entropy.
            for field in df.columns:
                results[field][k] = df[field].count(), df[field].mean(), df[field].std()
            

    def on_finish(self, results: Dict[str, pd.DataFrame]):
        model_type = self.model.replace(os.sep, "_")
        model_path = os.path.join(
            self._exp_cache_path(), f"{model_type}.pt",
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "field_entropies": self.field_entropies,
                    "row_entropies": self.row_entropies,
                    "model": self.model,
                },
                f,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=str, default=None, help="The experiments to run.")
    args = p.parse_args()

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


