# Pretraining and evaluation
import argparse
import os
from datetime import datetime

import lightning.pytorch as pl
import torch
import yaml
from lightning import Callback
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.profilers import PyTorchProfiler, AdvancedProfiler
from lightning.pytorch.tuner import Tuner
from transformers import GPT2Config, RwkvConfig

from model.model import EHRAuditGPT2, EHRAuditRWKV
from model.modules import EHRAuditPretraining, EHRAuditDataModule
from model.vocab import EHRVocab

__spec__ = None


class DebugCallback(Callback):
    def __init__(self):
        self.debug_now = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Model to use for pretraining."
    )
    parser.add_argument(
        "--max_epochs", type=int, default=5, help="Number of epochs to pretrain for."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size to use for pretraining."
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=1,
        help="Batches to wait before logging training progress.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether to profile the training process.",
    )
    parser.add_argument(
        "--dbg",
        action="store_true",
        help="Whether to run with single thread.",
    )
    parser.add_argument(
        "--reset_cache",
        action="store_true",
        help="Whether to reset the cache before training.",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use across train/val/test.",
    )
    parser.add_argument(
        "--conv_ckpt",
        type=str,
        default=None,
        help="Converts a Lightning checkpoint to a HuggingFace checkpoint.",
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
            n_head=6,
            n_layer=6,
        ),
    }

    dm = EHRAuditDataModule(
        config_path,
        vocab=vocab,
        batch_size=args.batch_size,
        reset_cache=args.reset_cache,
        debug=args.dbg,
    )

    model = models[args.model](model_configs[args.model], vocab)

    # Either load the model from a checkpoint for saving, or train it.
    if args.conv_ckpt is not None:
        pt_task = EHRAuditPretraining.load_from_checkpoint(args.conv_ckpt, model=model)
    else:
        pt_task = EHRAuditPretraining(model)
        profiler = None
        if args.profile:
            profiler = AdvancedProfiler(
                # sort_by_key="cpu_time",
                dirpath=os.path.normpath(os.path.join(path_prefix, config["log_path"])),
                filename="pt_profile",
            )

        train_max = (
            args.subset if args.profile is False else 100
        )  # If profiling is on, just use 100 batches
        val_max = args.subset if args.profile is False else 100
        test_max = args.subset if args.profile is False else 100

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger=pl.loggers.TensorBoardLogger(
                save_dir=os.path.normpath(
                    os.path.join(path_prefix, config["log_path"])
                ),
                name="pretraining",
            ),
            accumulate_grad_batches=32,
            profiler=profiler,
            limit_train_batches=train_max,
            limit_val_batches=val_max,
            limit_test_batches=test_max,
            callbacks=[TQDMProgressBar(refresh_rate=args.updates)],
        )

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
    param_count = sum(p.numel() for p in pt_task.model.parameters()) / 1e6
    todays_date = datetime.now().strftime("%Y-%m-%d")
    param_name = f"{args.model}/{param_count:.1f}M/{todays_date}".replace(".", "_")
    fname = os.path.normpath(
        os.path.join(path_prefix, config["pretrained_model_path"], param_name)
    )
    pt_task.model.save_pretrained(fname)
    print("Saved model to", fname)

    if args.conv_ckpt is None:
        print("Evaluating model")
        trainer.test(
            pt_task,
            datamodule=dm,
            verbose=True,
        )
