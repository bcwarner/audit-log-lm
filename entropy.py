# Evaluates the calculated entropy values for the test set.
import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
from matplotlib import pyplot as plt

from model.model import EHRAuditGPT2
from model.modules import EHRAuditPretraining, EHRAuditDataModule
from model.data import EHRAuditDataset
from model.vocab import EHRVocab
import tikzplotlib
import numpy as np

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=int, default=None, help="Model to use for pretraining."
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum batches to use."
    )
    parser.add_argument(
        "--exp_suffix",
        type=str,
        default=None,
        help="Suffix to add to the output file name.",
    )
    parser.add_argument(
        "--by_provider",
        action="store_true",
        help="Whether to calculate entropy by provider.",
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

    # Load the test dataset
    vocab = EHRVocab(
        vocab_path=os.path.normpath(os.path.join(path_prefix, config["vocab_path"]))
    )
    model = EHRAuditGPT2.from_pretrained(model_path, vocab=vocab)

    dm = EHRAuditDataModule(
        yaml_config_path=config_path,
        vocab=vocab,
        batch_size=1,  # Just one sample at a time
    )
    dm.setup()
    dl = dm.test_dataloader()

    if args.max_samples is None:
        print(f"Maximum number of samples to evaluate (0 for all, max = {len(dl)}):")
        max_samples = min(int(input(">>>")), len(dl))
    else:
        max_samples = args.max_samples

    window_size = 3 * 30  # 30 action window

    # Calculate the entropy values for the test set
    ce_values = []
    batches_seen = 0
    for batch in tqdm(dl, total=max_samples):
        input_ids, labels = batch
        # Sliding window over the sequence
        with torch.no_grad():
            # Find the index of the first -100
            first_pad_idx = (
                (labels.view(-1) == -100).nonzero(as_tuple=True)[0][0].item()
            )
            # Copy the labels and targets
            input_ids_c = torch.zeros_like(input_ids)
            labels_c = labels.clone()
            # Set the labels to -100, zero out the input_ids
            labels_c[:, :] = -100

            ce_current = []

            for i in range(first_pad_idx):
                # Set the ith label and input_id
                input_ids_c[:, i] = input_ids[:, i]
                labels_c[:, i] = labels[:, i]
                if i > 0:
                    labels_c[:, i - 1] = -100  # One token at a time

                if i >= window_size:
                    input_ids_c[:, i - window_size] = 0
                    labels_c[:, i - window_size] = -100

                # Calculate the cross entropy
                loss, _, _ = model(input_ids_c, labels=labels_c)
                ce_current.append(loss.item())

            ce_values.append(np.mean(ce_current))

        batches_seen += 1
        if max_samples != 0 and batches_seen >= max_samples:
            break

    # Print statistics about the entropy values
    stats = {
        "Mean CE": np.mean(ce_values),
        "Median CE": np.median(ce_values),
        "Max CE": np.max(ce_values),
        "Min CE": np.min(ce_values),
        "Std CE": np.std(ce_values),
        "Perplexity": np.mean(np.exp(ce_values)),
    }

    print(tabulate(stats.items(), headers=["Metric", "Value"]))

    # Plot the entropy values
    print(f"Plotting entropy values for {len(ce_values)} samples...")
    plt.hist(ce_values, bins=100)
    plt.title(f"Entropy Values for Test Set (N = {len(ce_values)})")
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.show()
    plt.savefig(
        os.path.normpath(
            os.path.join(
                path_prefix,
                config["results_path"],
                f"entropy_{len(ce_values)}_{args.exp_suffix}.png",
            )
        )
    )
    tikzplotlib.save(
        os.path.normpath(
            os.path.join(
                path_prefix,
                config["results_path"],
                f"entropy_{len(ce_values)}_{args.exp_suffix}.tex",
            )
        )
    )

    # Plot as perplexity
    print(f"Plotting perplexity values for {len(ce_values)} samples...")
    plt.clf()
    plt.hist(np.exp(ce_values), bins=100)
    plt.title(f"Perplexity Values for Test Set (N = {len(ce_values)})")
    plt.xlabel("Perplexity")
    plt.ylabel("Count")
    plt.show()
    plt.savefig(
        os.path.normpath(
            os.path.join(
                path_prefix, config["results_path"], f"perplexity_{len(ce_values)}.png"
            )
        )
    )
    tikzplotlib.save(
        os.path.normpath(
            os.path.join(
                path_prefix, config["results_path"], f"perplexity_{len(ce_values)}.tex"
            )
        )
    )
