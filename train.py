# Pretraining and evaluation
import argparse
import os

import lightning.pytorch as pl
import yaml
from transformers import GPT2Config, RwkvConfig

from model.model import EHRAuditGPT2, EHRAuditRWKV
from model.modules import EHRAuditPretraining, EHRAuditDataModule
from model.vocab import EHRVocab

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

vocab = EHRVocab(vocab_path=config["vocab_path"])
dm = EHRAuditDataModule(config_path)

models = {
    "gpt2": EHRAuditGPT2,
}
model_configs = {
    "gpt2": GPT2Config(
        vocab_size=len(vocab),
        n_positions=4096,
    ),
}
model = models[args.model](model_configs[args.model], vocab)

trainer = pl.Trainer(
    max_epochs=args.max_epochs,
)
trainer.fit(
    EHRAuditPretraining(model),
    datamodule=dm,
)
