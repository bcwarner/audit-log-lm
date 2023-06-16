# Used to build the vocab for the EHR audit log dataset.
import argparse
from collections import OrderedDict
import os
import pickle
from typing import Dict, List

import numpy as np
import yaml


class EHRVocab:
    def __init__(
        self,
        categorical_column_opts: Dict[str, List[str]] = None,
        vocab_path=None,
    ):
        """

        :param categorical_column_opts: Mapping of categorical column names to the list of possible values.
        :param max_len: Maximum length of the input sequence.
        :param vocab_path: Where to save/load the vocab.
        """
        if vocab_path is not None and os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f:
                self.__dict__.update(pickle.load(f).__dict__)
        else:
            # Load the other vocab options from the config file.
            self.field_tokens = OrderedDict()
            self.field_ids = OrderedDict()
            self.global_tokens = OrderedDict()

            # Set the default vocab options for GPT-style models.
            self.special_tokens = {
                "unk_token": "<unk>",
                "eos_token": "<eos>",
            }

            def new_token(field, value):
                if field not in self.field_tokens:
                    self.field_tokens[field] = OrderedDict()
                    self.field_ids[field] = []

                self.field_tokens[field][value] = len(self.global_tokens)
                self.field_ids[field].append(len(self.global_tokens))
                self.global_tokens[len(self.global_tokens)] = (field, value)

            # Allocate the base tokens.
            for k, v in self.special_tokens.items():
                setattr(self, k, v)
                new_token("special", v)

            self.vocab_path = vocab_path

            # Allocate the categorical tokens.
            for category, tokens in categorical_column_opts.items():
                for t in tokens:
                    new_token(category, t)

    def save(self):
        with open(self.vocab_path, "wb") as f:
            pickle.dump(self, f.__dict__)

    def field_to_token(self, field, value):
        return self.field_tokens[field][value]

    def token_to_global(self, token):
        return self.global_tokens[token]

    def global_to_token(self, global_id):
        return self.global_tokens[global_id][1]

    def globals_to_locals(self, global_ids):
        return [self.global_to_token(g) for g in global_ids]

    def field_names(self):
        return list(self.field_tokens.keys())

    def __len__(self):
        return len(self.global_tokens)


if __name__ == "__main__":
    # Load the config
    with open(os.path.join(os.path.pardir, "config.yaml")) as f:
        config = yaml.safe_load(f)

    # Erase the old vocab
    if os.path.exists(config["vocab_path"]):
        os.remove(config["vocab_path"])

    # This is where we'll actually build the vocab and then save it.
    categorical_column_opts = dict()

    # Build the patient IDs
    categorical_column_opts["PAT_ID"] = [
        str(i) for i in range(config["patient_id_max"])
    ]

    # Time deltas
    bins = getattr(np, config["timestamp_bins"]["spacing"])(
        config["timestamp_bins"]["min"],
        config["timestamp_bins"]["max"],
        config["timestamp_bins"]["bins"],
    )

    # Segfault otherwise
    import pandas as pd

    # METRIC_NAME
    df = pd.read_excel(config["metric_name_dict"]["file"], engine="openpyxl")
    categorical_column_opts["METRIC_NAME"] = df[
        config["metric_name_dict"]["column"]
    ].tolist()

    # Create the vocab
    vocab = EHRVocab(categorical_column_opts, vocab_path=config["vocab_path"])
    vocab.save()

    # Print the vocab
    print("Field tokens:")
    print(vocab.field_tokens)
    print("Field IDs:")
    print(vocab.field_ids)
    print("Global tokens:")
    print(vocab.global_tokens)
