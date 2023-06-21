# Evaluates the calculated entropy values for the test set.
import os
import yaml

from model.model import EHRAuditGPT2
from ..model.modules import EHRAuditPretraining
from ..model.data import EHRAuditDataset
from ..model.vocab import EHRVocab

if __name__ == "__main__":
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

    model_list = os.listdir(
        os.path.normpath(os.path.join(path_prefix, config["model_path"]))
    )
    print("Select a model to evaluate:")
    for i, model in enumerate(model_list):
        print(f"{i}: {model}")

    model_idx = int(input("Model index >>>"))
    model_name = model_list[model_idx]
    model_path = os.path.normpath(
        os.path.join(path_prefix, config["model_path"], model_name)
    )
    model = EHRAuditPretraining(
        EHRAuditGPT2.from_pretrained(model_path),
    )

    # Load the test dataset
    vocab = EHRVocab(
        vocab_path=os.path.normpath(os.path.join(path_prefix, config["vocab_path"]))
    )
