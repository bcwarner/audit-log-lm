# Prompt to push each variation of our models to the HuggingFace Hub.
import argparse
import os

import yaml
from huggingface_hub import ModelCardData, ModelCard
from transformers import PreTrainedModel

from model.model import EHRAuditGPT2, EHRAuditRWKV, EHRAuditLlama
from model.vocab import EHRVocab
import huggingface_hub

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

    api = huggingface_hub.HfApi()

    # Iterate through and push each model to the Hub
    for model_idx, model_name in enumerate(model_list):

        # Load the model
        model_path = os.path.normpath(os.path.join(model_paths, model_name))
        model_props = model_name.split(os.sep)
        model_type = model_props[0]
        model_params = model_props[1]
        model_date = model_props[2]
        hf_name = "/" + "-".join([config["huggingface"]["prefix"], model_type, model_params])
        hf_repo = config["huggingface"]["username"] + hf_name
        github_link =

        print(f"===== {hf_name} =====")
        desc = f"""
        # {hf_name}
        
        EHR audit logs are a highly granular stream of events that capture clinician activities, and is a 
        significant area of interest for research in characterizing clinician workflow on the electronic health record 
        (EHR). Existing techniques to measure the complexity of workflow through EHR audit logs (audit logs) involve 
        time- or frequency-based cross-sectional aggregations that are unable to capture the full complexity of a EHR 
        session. We briefly evaluate the usage of transformer-based tabular language model (tabular LM) in measuring 
        the entropy or disorderedness of action sequences within workflow and release the evaluated models publicly.
        
        The code to train our model is available on GitHub[The model was trained on a private dataset of ICU clinicians from Washington University in St. Louis
        
        # Usage 
        
        This model aims to be mostly compatible with the `transformers` library from HuggingFace. To use this model, you
        
        
        ## Entropy Calculation
        
        ## Generation
        
        Please cite our paper if you use this model in your work:
        ```
        [TBA]
        ```
        """

        should_push = input(f"Push {model_name} to HF as {hf_name}? (y/n): ").lower() == "y"

        if not should_push:
            continue

        model: PreTrainedModel = types[model_type].from_pretrained(model_path, vocab=vocab)
        model.push_to_hub(
            repo_id=hf_name,
            private=args.debug,
            commit_message=f"Uploading {hf_name}"
        )

        # Add the vocab to the repo
        api.upload_file(
            path_or_fileobj=os.path.normpath(os.path.join(path_prefix, config["vocab_path"])),
            path_in_repo=config["vocab_path"].split(os.sep)[-1],
            repo_id=hf_name,
            commit_message=f"Uploading vocab"
        )

        # Add the model card to the repo
        card_data = ModelCardData(
            license="apache-2.0",
            model_name=hf_name,
            tags=["tabular-regression", "ehr", "transformer"],
        )

        card = ModelCard.from_template(
            card_data=card_data,
            model_descrtiption=

        )
