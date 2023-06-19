# Pretraining and evaluation
import argparse
import os

import lightning.pytorch as pl
import yaml
from transformers import GPT2Config, RwkvConfig

from model.model import EHRAuditGPT2, EHRAuditRWKV
from model.modules import EHRAuditPretraining, EHRAuditDataModule
from model.vocab import EHRVocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Model to use for pretraining."
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Number of epochs to pretrain for."
    )
    args = parser.parse_args()

    # Load configuration and vocab
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    path_prefix = ""
    for prefix in config["path_prefix"]:
        if os.path.exists(prefix):
            path_prefix = prefix
            break

    vocab = EHRVocab(vocab_path=os.path.join(path_prefix, config["vocab_path"]))
    dm = EHRAuditDataModule(config_path, vocab=vocab)

    models = {
        "gpt2": EHRAuditGPT2,
    }
    model_configs = {
        "gpt2": GPT2Config(
            vocab_size=len(vocab),
            n_positions=1024,
            n_head=12,
            n_layer=6,
        ),
    }
    model = models[args.model](model_configs[args.model], vocab)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
    )
    pt_task = EHRAuditPretraining(model)

    trainer.fit(
        pt_task,
        datamodule=dm,
    )

    # Save the model according to the HuggingFace API
    print("Saving model to", os.path.join(path_prefix, config["pretrained_model_path"]))
    pt_task.model.save_pretrained(
        os.path.join(path_prefix, config["pretrained_model_path"])
    )

    print("Evaluating model")
    trainer.test(
        pt_task,
        datamodule=dm,
        verbose=True,
    )
