# Collects all cached versions of the entropy data and runs experiments atop them.
import inspect
import os
import pickle
import sys

import joblib
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
        #vocab: EHRVocab,
        #model: str,
        *args,
        **kwargs,
    ):
        self.config = config
        self.path_prefix = path_prefix
        #self.vocab = vocab
        #self.model = model

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
                f"exp_cache_{self.__class__.__name__}.pkl",
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
    def __init__(self, config, path_prefix,
                 #vocab, model,
                 *args, **kwargs):
        super().__init__(config, path_prefix,
                         #vocab, model,
                         *args, **kwargs)
        self.field_entropies = defaultdict(list)
        self.row_entropies = []
        self._samples_seen = 0

    def requirements(self):
        return {
            "logs": False,
            "provider_aware": True,
            "provider_unaware": True,
        }

    def map(self,
            provider=None,
            audit_log_df: pd.DataFrame = None,
            provider_aware_df: Dict[str, pd.DataFrame] = None,
            provider_unaware_df: Dict[str, pd.DataFrame] = None,
            ):
        # field => model => entropy count, average, std
        results = defaultdict(lambda: defaultdict(list))
        for k, df in {**provider_aware_df, **provider_unaware_df}.items():
            # Iterate each of the fields in the df and aggregate the entropy.
            for field in df.columns:
                results[field][k] = (df[field].count(), df[field].mean())
        return results
            

    def on_finish(self, results: Dict[str, object]):
        entropy_scores = defaultdict(lambda: defaultdict(lambda: (0, 0)))
        provider_aware = set()
        provider_unaware = set()
        for provider, result in results.items():
            # Merge each of the results from each provider by field => provider, adjusting mean and std.
            for field, models in result.items():
                if "Unnamed: 0" in field:
                    continue
                for model, (new_count, new_mean) in models.items():
                    prev_count, prev_mean = entropy_scores[model][field]
                    entropy_scores[model][field] = (
                        prev_count + new_count,
                        (prev_mean * prev_count + new_mean * new_count) / (prev_count + new_count),
                    )
                    if "USER_ID" in result.keys():
                        provider_aware.add(field)
                    else:
                        provider_unaware.add(field)


        self.field_entropies = entropy_scores
        self.provider_aware = provider_aware
        self.provider_unaware = provider_unaware

        # Convert the nested defaultdicts to dicts.
        self.field_entropies = {k: dict(v) for k, v in self.field_entropies.items()}
        self.provider_aware = list(self.provider_aware)
        self.provider_unaware = list(self.provider_unaware)

        with open(self._exp_cache_path(), "wb") as f:
            pickle.dump(
                {
                    "field_entropies": self.field_entropies,
                    "provider_aware": self.provider_aware,
                    "provider_unaware": self.provider_unaware,
                },
                f,
            )

    def plot(self):
        with open(self._exp_cache_path(), "rb") as f:
            results = pickle.load(f)
            self.field_entropies = results["field_entropies"]
            self.provider_aware = results["provider_aware"]
            self.provider_unaware = results["provider_unaware"]

        model_count = len(self.provider_unaware) + len(self.provider_aware)
        model_count_unaware = len(self.provider_unaware)

        # Convert the nested default dict to a dict
        self.field_entropies = {k: defaultdict(lambda: (0, 0), v) for k, v in self.field_entropies.items()}

        # Plot the field entropies
        plt.clf()
        fig, ax = plt.gcf(), plt.gca()
        width = 0.1
        field_labels = ["METRIC_NAME", "PAT_ID", "ACCESS_TIME"]
        field_labels_aware = field_labels + ["USER_ID"]
        range = np.arange(len(field_labels_aware))
        max_ht = 0
        sorted_keys = sorted(self.field_entropies.keys())
        for idx, key in enumerate(sorted_keys):
            key_nice = key.replace("entropy-", "").replace("_", ".").replace(".csv", "")
            if "USER_ID" in self.field_entropies[key].keys():
                hts = [2 ** np.mean(self.field_entropies[key][k][1]) for k in field_labels_aware]
                max_ht = max(max_ht, max(hts))
                key_nice += " (PA)"
                rects = ax.barh(range + (idx * width), height=width, width=hts, label=key_nice)
                ax.bar_label(rects, fmt="%.4f")
            else:
                hts = [2 ** np.mean(self.field_entropies[key][k][1]) for k in field_labels]
                max_ht = max(max_ht, max(hts))
                rects = ax.barh(range[:-1] + (idx * width), height=width, width=hts, label=key_nice)
                ax.bar_label(rects, fmt="%.4f")

        ax.set_xlim(0, 1.25 * max_ht)
        ax.set_yticks(range + (width * model_count) / 2, field_labels_aware)
        ax.set_xlabel("Perplexity")
        ax.set_title("Perplexity by Field")
        fig.tight_layout()

        plt.legend()
        plt.savefig(
            os.path.normpath(
                os.path.join(
                    self.path_prefix,
                    self.config["results_path"],
                    "field_entropies.pdf",
                )
            )
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=str, default="all", help="The experiments to run.")
    p.add_argument("--plot", type=str, default="", help="The experiments to plot.")
    p.add_argument("--debug", action="store_true", help="Whether to run in debug mode.")
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

    # Load the data module and pull out the providers from there for the provider unaware models.from
    # All providers will be used (except for the one without one from the exclusion list).
    # Initialize each of the experiments desired
    if "," in args.exp:
        experiments = [
            eval(exp)(config, path_prefix) for exp in args.exp.split(",")
        ]
    elif "all" in args.exp:
        # Get a list of all classes that sublcass Experiment in this file.
        exp_classes = [
            obj
            for name, obj in inspect.getmembers(sys.modules[__name__])
            if inspect.isclass(obj)
               and issubclass(obj, Experiment)
               and obj != Experiment
        ]
        experiments = [exp(config, path_prefix) for exp in exp_classes]
    else:
        experiments = [eval(args.exp)(config, path_prefix)]

    if args.plot == "only":
        for exp in experiments:
            exp.plot()
        sys.exit()

    # Determine what we need to load for all of the experiments.
    requirements = defaultdict(bool)
    for exp in experiments:
        for k, v in exp.requirements().items():
            requirements[k] = requirements[k] or v

    # Get the list of providers in the data directory.
    data_path = os.path.normpath(
        os.path.join(path_prefix, config["audit_log_path"])
    )

    def dispatch_map(provider):
        # Double check that the audit log is non-empty
        log_path = os.path.normpath(os.path.join(data_path, provider, config["audit_log_file"]))
        if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
            return None

        audit_log_df = None
        provider_aware_df = []
        provider_unaware_df = []

        if requirements["logs"]:
            audit_log_df = pd.read_csv(log_path, sep=",")

        # Iterate through all files prefixed with entropy, and split them into lists of provider aware and provider unaware.
        for file in os.listdir(os.path.normpath(os.path.join(data_path, provider))):
            if file.startswith("entropy"):
                # Read the first line of the file to determine if it's provider aware or not.
                with open(os.path.normpath(os.path.join(data_path, provider, file)), "r") as f:
                    first_line = f.readline()
                    if "USER_ID" in first_line:
                        provider_aware_df.append(file)
                    else:
                        provider_unaware_df.append(file)

        # Load the provider aware and provider unaware dataframes as desired.
        if requirements["provider_aware"]:
            provider_aware_df = {file: pd.read_csv(os.path.normpath(os.path.join(data_path, provider, file)), sep=",") for file in provider_aware_df}

        if requirements["provider_unaware"]:
            provider_unaware_df = {file: pd.read_csv(os.path.normpath(os.path.join(data_path, provider, file)), sep=",") for file in provider_unaware_df}

        # Iterate through each of the experiments and run the map function.
        results = {}
        for exp in experiments:
            print(f"Running {exp.__class__.__name__} on {provider}")
            results[exp.__class__.__name__] = exp.map(
                provider=provider,
                audit_log_df=audit_log_df,
                provider_aware_df=provider_aware_df,
                provider_unaware_df=provider_unaware_df,
            )

        return results

    # Dispatch jobs for each proivder in the directory
    providers = os.listdir(data_path)
    exp_results = joblib.Parallel(n_jobs=-1 if not args.debug else 1, verbose=2)(
        joblib.delayed(dispatch_map)(provider) for provider in providers
    )

    # Iterate through each of the experiments and run the on_finish function.
    for exp in experiments:
        exp.on_finish({p: r[exp.__class__.__name__] for p, r in zip(providers, exp_results) if r is not None})

    # Then plot
    for exp in experiments:
        exp.plot()