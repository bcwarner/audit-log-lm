# Prompt to push each variation of our models to the HuggingFace Hub.
import argparse
import os

import yaml
from transformers import PreTrainedModel

from model.model import EHRAuditGPT2, EHRAuditRWKV, EHRAuditLlama
from model.vocab import EHRVocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dry run, don't actually push to HF",
    )
    args = parser.parse_args()

    # Load configuration and vocab
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

    if len(model_list) == 0:
        raise ValueError(f"No models found in {format(model_paths)}")

    vocab = EHRVocab(
        vocab_path=os.path.normpath(os.path.join(path_prefix, config["vocab_path"]))
    )
    types = {
        "gpt2": EHRAuditGPT2,
        "rwkv": EHRAuditRWKV,
        "llama": EHRAuditLlama,
    }

    # Iterate through and push each model to the Hub
    for model_idx, model_name in enumerate(model_list):

        # Load the model
        model_path = os.path.normpath(os.path.join(model_paths, model_name))
        model_props = model_name.split(os.sep)
        model_type = model_props[0]
        model_params = model_props[1]
        model_date = model_props[2]
        hf_name = config["huggingface"]["username"] + "/" + "-".join([config["huggingface"]["prefix"], model_type, model_params])

        should_push = input(f"Push {model_name} to HF as {hf_name}? (y/n): ").lower() == "y"
        if not should_push:
            continue

        model: PreTrainedModel = types[model_type].from_pretrained(model_path, vocab=vocab)
        model.push_to_hub(
            repo_id=hf_name,
            private=args.debug,
            commit_message=f"Uploading {hf_name}"
        )


