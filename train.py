# Pretraining and evaluation
import argparse
import os

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner import Tuner
from transformers import GPT2Config, RwkvConfig

from model.model import EHRAuditGPT2, EHRAuditRWKV
from model.modules import EHRAuditPretraining, EHRAuditDataModule
from model.vocab import EHRVocab

__spec__ = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Model to use for pretraining."
    )
    parser.add_argument(
        "--max_epochs", type=int, default=5, help="Number of epochs to pretrain for."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size to use for pretraining."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether to profile the training process.",
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

    vocab = EHRVocab(
        vocab_path=os.path.normpath(os.path.join(path_prefix, config["vocab_path"]))
    )
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

    dm = EHRAuditDataModule(
        config_path,
        vocab=vocab,
        batch_size=args.batch_size,
    )

    model = models[args.model](model_configs[args.model], vocab)

    profiler = None
    if args.profile:
        profiler = PyTorchProfiler(
            sort_by_key="cpu_time",
            dirpath=os.path.normpath(os.path.join(path_prefix, config["log_path"])),
            filename="pt_profile",
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=os.path.normpath(os.path.join(path_prefix, config["log_path"])),
            name="pretraining",
        ),
        # devices=0 if (os.name == "nt" or not torch.has_cuda) else -1,
        accumulate_grad_batches=1024,
        profiler=profiler,
        limit_train_batches=100 if args.profile else None,
    )

    pt_task = EHRAuditPretraining(model)
    # if args.strategy != "ddp":
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(pt_task, datamodule=dm)

    trainer.fit(
        pt_task,
        datamodule=dm,
    )

    if trainer.interrupted and trainer.current_epoch == 0:
        # Allow for model saves if we did past one epoch.
        exit(0)

    # Save the model according to the HuggingFace API
    if path_prefix == "/storage1/" and not args.profile:
        # Second safeguard against overwriting a model.
        param_count = sum(p.numel() for p in pt_task.model.parameters()) / 1e6
        param_name = f"{args.model}/{param_count:.1f}M".replace(".", "_")
        fname = os.path.normpath(
            os.path.join(path_prefix, config["pretrained_model_path"], param_name)
        )
        pt_task.model.save_pretrained(fname)

    print("Evaluating model")
    trainer.test(
        pt_task,
        datamodule=dm,
        verbose=True,
    )
