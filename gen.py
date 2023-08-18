# Demo that takes a random audit log sequence and predicts the next set of actions.
import argparse
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
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
from matplotlib import pyplot as plt
from transformers import BeamSearchScorer, StoppingCriteriaList, MaxLengthCriteria, LogitsProcessorList

from model.model import EHRAuditGPT2
from model.modules import EHRAuditPretraining, EHRAuditDataModule
from model.data import timestamp_space_calculation
from model.vocab import EHRVocab, EHRAuditTokenizer, EHRAuditLogitsProcessor
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
        "--start",
        type=int,
        default=0,
        help="Index of the first audit log to use for the demo.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of audit logs to use for the demo.",
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

    if args.model is None:
        print("Select a model to evaluate:")
        for i, model in enumerate(sorted(model_list)):
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
    model = EHRAuditGPT2.from_pretrained(model_path, vocab=vocab)
    model.to(device)

    dm = EHRAuditDataModule(
        yaml_config_path=config_path,
        vocab=vocab,
        batch_size=1,  # Just one sample at a time
    )
    dm.setup()
    if args.val:
        dl = dm.val_dataloader()
    else:
        dl = dm.test_dataloader()

    tk = EHRAuditTokenizer(vocab)

    window_size = 30  # 30 action window
    predict_size = window_size * 2  # input + output
    proc_count = 0
    for batch in dl:
        input_ids, labels = batch
        with torch.no_grad():
            # Find the eos index
            nonzeros = (labels.view(-1) == -100).nonzero(as_tuple=True)
            if len(nonzeros[0]) == 0:
                eos_index = len(labels.view(-1)) - 1
            else:
                eos_index = nonzeros[0][0].item() - 1

            ce_current = []

            row_len = len(vocab.field_ids) - 1  # Exclude special fields
            row_count = (eos_index - 1) // row_len
            if row_count <= predict_size // row_len:  # Not applicable
                continue

            # Copy the labels, and zero out past the window length.
            input_ids_c = input_ids.clone()
            input_ids_c = input_ids_c[:, :(window_size + 1)]

            # Stack them to the beam size
            # beam_size = 1
            # input_ids_c = input_ids_c.repeat(beam_size, input_ids.size(1)).to(device)
            #
            # beam_scorer = BeamSearchScorer(
            #     batch_size=1,
            #     num_beams=beam_size,
            #     device=device,
            # )

            # Stopping criteria, just the second window
            sc_list = StoppingCriteriaList([MaxLengthCriteria(predict_size)])

            # Process logits so that only tokens in the correct field are allowed
            logits_processor = LogitsProcessorList(
                [
                    EHRAuditLogitsProcessor(vocab=vocab),
                ]
            )

            if isinstance(model, EHRAuditGPT2):
                model.generation_config.pad_token_id = model.generation_config.eos_token_id

            # Generate the outputs from the window.
            outputs = model.greedy_search(
                input_ids_c.to(device),
                stopping_criteria=sc_list,
                logits_processor=logits_processor,
            )

            # Decode the outputs
            predictions: DataFrame = tk.decode(outputs[0].cpu().numpy())
            full_input: DataFrame = tk.decode(input_ids[0, :predict_size].cpu().numpy())

            # Change the indicies of predictions and full_input to a multi-index
            full_input.columns = pd.MultiIndex.from_tuples(
                [("Ground-Truth", x) for x in full_input.columns]
            )
            predictions.columns = pd.MultiIndex.from_tuples(
                [("Prediction", x) for x in predictions.columns]
            )

            # Concatenate the two dataframes so that they're side by side
            full_df = pd.concat([full_input, predictions], axis=1)

            # Add a column for displaying ground-truth or predictions.
            full_df["Type"] = ["Input"] * (window_size // row_len) + ["Prediction"] * (window_size // row_len)

            with pd.option_context("display.max_columns", None, "display.width", None):
                print(full_df)
        proc_count += 1
        if proc_count >= args.count:
            break
