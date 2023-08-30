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

class GenerationExperiment:
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

    def stopping_criteria(self, context_length: int = 0):
        return None

    def eval_generation(self, output_df, label_df):
        pass

    def _exp_cache_path(self):
        return os.path.normpath(
            os.path.join(
                self.path_prefix,
                self.config["results_path"],
                f"exp_cache_{self.__class__.__name__}",
            )
        )

    def on_finish(self):
        pass

    def plot(self):
        pass

    # None = all preceding, # = fixed preceding
    def window_size(self):
        return None

    def min_size(self):
        return 2

    def examples_seen(self):
        return 0

# Next whole-action prediction
# Next METRIC_NAME prediction
class NextActionExperiment(GenerationExperiment):
    def __init__(self, config: dict,
                path_prefix: str,
                vocab: EHRVocab,
                model: str,
                *args,
                **kwargs):
        super().__init__(config, path_prefix, vocab, model, *args, **kwargs)
        self.total_seen = 0
        self.correct_by_field = defaultdict(int)
        self.total_correct = 0

    def window_size(self):
        return 1

    def stopping_criteria(self, context_length: int = 0):
        return StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=context_length + 3),
            ]
        )

    def eval_generation(self, output_df, label_df):
        # Get the next action
        next_action = label_df.iloc[0]
        # Get the predicted next action
        predicted_next_action = output_df.iloc[0]
        # Check if the predicted next action is correct
        for x in [METRIC_NAME_COL, PAT_ID_COL, ACCESS_TIME_COL]:
            if next_action[x] == predicted_next_action[x]:
                self.correct_by_field[x] += 1
        if np.all(next_action == predicted_next_action):
            self.total_correct += 1
        self.total_seen += 1

    def examples_seen(self):
        return self.total_seen

    def on_finish(self):
        model_type = self.model.replace(os.sep, "_")
        model_path = os.path.join(
            self._exp_cache_path(), f"{model_type}.pkl"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "total_seen": self.total_seen,
                    "correct_by_field": self.correct_by_field,
                    "total_correct": self.total_correct,
                },
                f,
            )

    def plot(self):
        model_versions = os.listdir(self._exp_cache_path())
        total_seen_by_model = defaultdict(int)
        metric_correct_by_model = defaultdict(int)
        pat_id_correct_by_model = defaultdict(int)
        access_time_correct_by_model = defaultdict(int)
        total_correct_by_model = defaultdict(int)

        # Convert to a LaTeX table
        rows = []
        for model_version in model_versions:
            model_path = os.path.join(
                self._exp_cache_path(), model_version
            )
            model_fs = model_version.split("_")
            model_t = model_fs[0]
            model_type = model_t + "-" + ".".join(model_fs[1:3])

            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            total_seen_by_model[model_type] = model_data["total_seen"]
            metric_correct_by_model[model_type] = model_data["correct_by_field"][METRIC_NAME_COL]
            pat_id_correct_by_model[model_type] = model_data["correct_by_field"][PAT_ID_COL]
            access_time_correct_by_model[model_type] = model_data["correct_by_field"][ACCESS_TIME_COL]
            total_correct_by_model[model_type] = model_data["total_correct"]

            rows.append({
                "Model": model_type,
                "Total Seen": model_data["total_seen"],
                "METRIC_NAME Accuracy": metric_correct_by_model[model_type] / total_seen_by_model[model_type],
                "PAT_ID Accuracy": pat_id_correct_by_model[model_type] / total_seen_by_model[model_type],
                "ACCESS_TIME Accuracy": access_time_correct_by_model[model_type] / total_seen_by_model[model_type],
                "Total Accuracy": total_correct_by_model[model_type] / total_seen_by_model[model_type],
            })
        df = pd.DataFrame(rows)

        df = df.sort_values(by=["METRIC_NAME Accuracy"], ascending=True)
        # Drop the Total Seen column if all values are the same
        if len(df["Total Seen"].unique()) == 1:
            df = df.drop(columns=["Total Seen"])

        out = df.to_latex(index=False,
                    float_format="%.3f",
                    caption="Results for predicting the next action in an audit log sequence.",
                    label="tab:next_action_results")

        print(out)



# Next correct METRIC\_NAME takes how many actions?
class CorrectAppearance(GenerationExperiment):
    def __init__(self, vocab=None, model=None):
        super().__init__(vocab=vocab, model=model)
        self.total_seen = 0


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
        default=0,
        help="Number of audit logs to use for the demo.",
    )
    parser.add_argument(
        "--search",
        "-s",
        type=str,
        default="greedy",
        help="Search method to use for decoding. If beam, :k is the beam size.",
    )
    parser.add_argument(
        "-p",
        "--plot",
        type=str,
        default="no", # other options: "yes", "only"
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="NextActionExperiment",
        help="Experiment to run.",
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

    tk = EHRAuditTokenizer(vocab, timestamp_spaces_cal=timestamp_space_calculation([
                    config["timestamp_bins"]["spacing"],
                    config["timestamp_bins"]["min"],
                    config["timestamp_bins"]["max"],
                    config["timestamp_bins"]["bins"],
                ]))

    exp = eval(args.exp)(vocab=vocab, model=model_name, config=config, path_prefix=path_prefix)
    if args.plot == "only":
        exp.plot()
        sys.exit(0)

    row_len = len(vocab.field_ids) - 1  # Exclude special fields

    max_samples = min(len(dl), args.count) if args.count > 0 else len(dl)
    pbar = tqdm(total=max_samples, desc=f"Examples Seen For {exp.__class__.__name__}")
    for idx, batch in enumerate(dl):
        if idx < args.start:
            continue

        input_ids, labels = batch
        with torch.no_grad():
            # Find the eos index
            nonzeros = (labels.view(-1) == -100).nonzero(as_tuple=True)
            if len(nonzeros[0]) == 0:
                eos_index = len(labels.view(-1)) - 1
            else:
                eos_index = nonzeros[0][0].item() - 1

            row_count = (eos_index - 1) // row_len
            if row_count <= exp.window_size() // row_len:  # Not applicable
                continue
            for i in range(1, row_count):
                window_size = i * row_len

                # Copy the labels, and zero out past the window length.
                input_ids_c = torch.zeros((input_ids.shape[0], window_size)).long()
                input_ids_c[:, :window_size] = input_ids[:, :window_size]

                # Stack them to the beam size

                # Stopping criteria
                sc_list = exp.stopping_criteria(window_size)

                # Process logits so that only tokens in the correct field are allowed
                logits_processor = LogitsProcessorList(
                    [
                        EHRAuditLogitsProcessor(vocab=vocab),
                    ]
                )

                if isinstance(model, EHRAuditGPT2):
                    model.generation_config.pad_token_id = model.generation_config.eos_token_id

                # Generate the outputs from the window.
                if args.search == "greedy":
                    outputs = model.greedy_search(
                        input_ids_c.to(device),
                        stopping_criteria=sc_list,
                        logits_processor=logits_processor,
                    )
                elif args.search == "sample":
                    outputs = model.sample(
                        input_ids_c.to(device),
                        stopping_criteria=sc_list,
                        logits_processor=logits_processor,
                    )
                elif "contrastive" in args.search:
                    opts = args.search.split(":")
                    if len(opts) == 1:
                        opts.append("5")
                    if len(opts) <= 2:
                        opts.append("0.1")
                    outputs = model.contrastive_search(
                        input_ids_c.to(device),
                        stopping_criteria=sc_list,
                        logits_processor=logits_processor,
                        top_k=int(opts[1]),
                        penalty_alpha=float(opts[2]),
                    )
                elif "beam" in args.search:
                    # Errors out with CUBLAS_STATUS_NOT_INITIALIZED, not sure why
                    beam_size = int(args.search.split(":")[1])
                    input_ids_c = input_ids_c.repeat(beam_size, input_ids.size(1)).to(device)

                    beam_scorer = BeamSearchScorer(
                        batch_size=1,
                        num_beams=beam_size,
                        device=device,
                    )

                    outputs = model.beam_search(
                        input_ids_c.to(device),
                        stopping_criteria=sc_list,
                        logits_processor=logits_processor,
                        beam_scorer=beam_scorer,
                    )

                # Decode the outputs
                predictions: DataFrame = tk.decode(outputs[0, window_size:].cpu().numpy())
                labels: DataFrame = tk.decode(input_ids[0, window_size:].cpu().numpy())

                exp.eval_generation(predictions, labels)
                pbar.n = exp.examples_seen()
                pbar.refresh()
                """
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
                full_df["Type"] = ["Input"] * (window_size // row_len) + ["Prediction"] * args.predict_size

                with pd.option_context("display.max_columns", None, "display.max_rows", None, "display.width", None):
                    print(f"==== Predictions for Example {idx} ====")
                    print(full_df)
                """
        if exp.examples_seen() >= max_samples:
            break

    pbar.close()
    exp.on_finish()
    if args.plot == "yes":
        exp.plot()
