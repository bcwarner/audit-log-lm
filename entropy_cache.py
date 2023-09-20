# Assigns entropy values with a given model to the dataset in the order it appears.
import argparse
import bisect
import inspect
import os
import pickle
import sys
from collections import defaultdict

import pandas as pd
import scipy.stats
import torch
import yaml
from matplotlib.axes import Axes
from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
from matplotlib import pyplot as plt

from model.model import EHRAuditGPT2, EHRAuditRWKV, EHRAuditLlama
from model.modules import EHRAuditPretraining, EHRAuditDataModule
from model.data import timestamp_space_calculation
from model.vocab import EHRVocab, EHRAuditTokenizer
import tikzplotlib
import numpy as np

# Fyi: this is a quick-and-dirty way of id'ing the columns, will need to be changed if the tabularization changes
METRIC_NAME_COL = 0
PAT_ID_COL = 1
ACCESS_TIME_COL = 2


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=int, default=None, help="Model to use for pretraining."
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Run with the validation dataset instead of the test.",
    )
    parser.add_argument(
        "--reset_cache",
        action="store_true",
        help="Whether to reset the cache before analysis.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run with single thread.",
    )
    args = parser.parse_args()
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

    model_list = sorted(model_list)
    if args.model is None:
        print("Select a model to evaluate:")
        for i, model in enumerate(model_list):
            print(f"{i}: {model}")

        model_idx = int(input("Model index >>>"))
    else:
        model_idx = args.model

    model_name = model_list[model_idx]
    model_path = os.path.normpath(
        os.path.join(path_prefix, config["pretrained_model_path"], model_name)
    )

    # Get the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    vocab = EHRVocab(
        vocab_path=os.path.normpath(os.path.join(path_prefix, config["vocab_path"]))
    )

    dm = EHRAuditDataModule(
        yaml_config_path=config_path,
        vocab=vocab,
        batch_size=1,  # Just one sample at a time
        reset_cache=args.reset_cache,
        debug=args.debug,
    )
    if args.reset_cache:
        dm.prepare_data()
    dm.setup()

    types = {
        "gpt2": EHRAuditGPT2,
        "rwkv": EHRAuditRWKV,
        "llama": EHRAuditLlama,
    }

    model_type = model_list[model_idx].split(os.sep)[0]
    model = types[model_type].from_pretrained(model_path, vocab=vocab)
    model.loss.reduction = "none"
    model.to(device)

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()

    train_dl.shuffle = False
    val_dl.shuffle = False
    test_dl.shuffle = False

    def iter_dl(dl):
        # Provider => row => field => entropy
        whole_set_entropy_map = defaultdict(lambda:
                                            defaultdict(lambda:
                                                        {"METRIC_NAME": pd.NA, "PAT_ID": pd.NA, "ACCESS_TIME": pd.NA})
                                            )
        for batch_idx, batch in tqdm(enumerate(dl), total=len(dl)):
            input_ids, labels = batch

            with torch.no_grad():
                # Find the eos index
                nonzeros = (labels.view(-1) == -100).nonzero(as_tuple=True)
                if len(nonzeros[0]) == 0:
                    eos_index = len(labels.view(-1)) - 1
                else:
                    eos_index = nonzeros[0][0].item() - 1

                # Copy the labels and targets
                input_ids_c = torch.zeros_like(input_ids)
                labels_c = labels.clone()
                # Set the labels to -100, zero out the input_ids
                labels_c[:, :] = -100

                ce_current = []
                row_len = len(vocab.field_ids) - 1  # Exclude special fields
                row_count = (eos_index - 1) // row_len
                if row_count <= 1:  # Not applicable
                    continue

                # NOTE: Next-token generation != next-row generation
                # This means that we include the next two tokens in the input to avoid EOS predictions.
                loss_pos = model.loss.col_ids_labels.transpose(0, 1).flatten()

                # Get the index of the current row in the whole df
                dset_idx = bisect.bisect_right(dl.dataset.cumulative_sizes, batch_idx)
                dset = dl.dataset.datasets[dset_idx]
                provider = dset.provider

                for i in range(0, row_count):
                    input_ids_start = i * row_len
                    input_ids_end = input_ids_start + row_len
                    input_ids_end_extra = input_ids_end + row_len
                    # Get the current row
                    input_ids_c[:, input_ids_start:input_ids_end_extra] = input_ids[
                        :, input_ids_start:input_ids_end_extra
                    ]
                    # Labels are next row.
                    labels_row_start = (i + 1) * row_len
                    labels_row_end = labels_row_start + row_len
                    labels_c[:, labels_row_start:labels_row_end] = labels[
                        :, labels_row_start:labels_row_end
                    ]
                    #if i > 0:
                    #    labels_c[
                    #        :, input_ids_start:input_ids_end
                    #    ] = -100  # Eliminate previous row.

                    # if i >= window_size:
                    #    old_row_start = (i - window_size) * row_len
                    #    old_row_end = old_row_start + row_len
                    #    input_ids_c[:, old_row_start:old_row_end] = 0

                    # Calculate the cross entropy
                    output = model(input_ids_c.to(device), labels=labels_c.to(device), return_dict=True)
                    loss = output.loss.cpu().numpy()
                    metric_loss = loss[METRIC_NAME_COL, i]
                    patient_loss = loss[PAT_ID_COL, i]
                    time_loss = loss[ACCESS_TIME_COL, i]

                    whole_row_idx = dl.dataset.seqs_indices[dset_idx] + i

                    whole_set_entropy_map[provider][whole_row_idx]["METRIC_NAME"] = metric_loss
                    whole_set_entropy_map[provider][whole_row_idx]["PAT_ID"] = patient_loss
                    whole_set_entropy_map[provider][whole_row_idx]["ACCESS_TIME"] = time_loss




