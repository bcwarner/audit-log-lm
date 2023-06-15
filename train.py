# Pretraining and evaluation
import argparse

import lightning.pytorch as pl
from transformers import GPT2Config, RwkvConfig

from model.model import EHRAuditGPT2, EHRAuditRWKV
from model.modules import EHRAuditPretraining, EHRAuditDataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="rwkv", help="Model to use for pretraining."
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Number of epochs to pretrain for."
    )
    args = parser.parse_args()

    models = {
        "rwkv": EHRAuditRWKV,
        "gpt2": EHRAuditGPT2,
    }
    model_configs = {
        "rwkv": RwkvConfig(),
        "gpt2": GPT2Config(),
    }
    model = models[args.model](model_configs[args.model])
    dm = EHRAuditDataModule()
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
    )
    trainer.fit(
        EHRAuditPretraining(model, model_configs[args.model]),
        datamodule=EHRAuditDataModule(),
    )
