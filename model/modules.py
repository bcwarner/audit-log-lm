import os
from functools import partial
from typing import Any

import joblib
import torch
import yaml
from lightning import pytorch as pl
from torch.utils.data import ConcatDataset, random_split, ChainDataset
from tqdm import tqdm

from Sophia.sophia import SophiaG
from model.data import EHRAuditDataset
from model.vocab import EHRVocab


def collate_fn(batch, n_positions=1024):
    input_ids_col = []
    labels_col = []
    for input_ids in batch:
        pad_pos_count = n_positions - input_ids.size(0)
        # Pad to max length or crop to max length
        if input_ids.size(0) < n_positions:
            input_ids = torch.nn.functional.pad(
                input=input_ids,
                pad=(0, pad_pos_count),
                value=-100,  # EOS token
            )
        elif input_ids.size(0) > n_positions:
            input_ids = input_ids[:n_positions]

        labels = input_ids.clone().detach()
        input_ids[labels == -100] = 0

        input_ids_col.append(input_ids)
        labels_col.append(labels)

    return torch.stack(input_ids_col), torch.stack(labels_col)


def worker_fn(worker_id, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EHRAuditPretraining(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        self.step = 0

    def forward(self, input_ids, labels, should_break=False):
        return self.model(input_ids, labels=labels, should_break=should_break)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(*batch, should_break=False)
        loss = outputs[0]
        self.log(
            "train_loss",
            loss.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(*batch)
        loss = outputs[0]
        self.log(
            "val_loss",
            loss.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.forward(*batch)
        loss = outputs[0]
        self.log(
            "test_loss",
            loss.mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model(batch)

    def configure_optimizers(self):
        return SophiaG(
            self.model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )


class EHRAuditDataModule(pl.LightningDataModule):
    def __init__(
        self,
        yaml_config_path: str,
        vocab: EHRVocab,
        batch_size=1,
        n_positions=1024,
        reset_cache=False,
        debug=False,
    ):
        super().__init__()
        with open(yaml_config_path) as f:
            self.config = yaml.safe_load(f)
        self.vocab = vocab
        self.batch_size = batch_size
        self.n_positions = n_positions
        self.reset_cache = reset_cache
        self.debug = debug

    def prepare_data(self):
        # Itereate through each prefix and determine which one exists, then choose that one.
        path_prefix = ""
        for prefix in self.config["path_prefix"]:
            if os.path.exists(prefix):
                path_prefix = prefix
                break

        data_path = os.path.normpath(
            os.path.join(path_prefix, self.config["audit_log_path"])
        )
        log_name = self.config["audit_log_file"]
        shift_sep_min = self.config["sep_min"]["shift"]
        session_sep_min = self.config["sep_min"]["session"]

        def log_load(self, provider: str):
            prov_path = os.path.normpath(os.path.join(data_path, provider))
            # Check the file is not empty and exists, there's a couple of these.
            log_path = os.path.normpath(os.path.join(prov_path, log_name))
            if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                return

            # Skip datasets we've already prepared.
            if (
                os.path.exists(
                    os.path.normpath(
                        os.path.join(prov_path, self.config["audit_log_cache"])
                    )
                )
                and self.reset_cache is False
            ):
                return

            dset = EHRAuditDataset(
                prov_path,
                session_sep_min=session_sep_min,
                shift_sep_min=shift_sep_min,
                log_name=log_name,
                vocab=self.vocab,
                timestamp_spaces=[
                    self.config["timestamp_bins"]["spacing"],
                    self.config["timestamp_bins"]["min"],
                    self.config["timestamp_bins"]["max"],
                    self.config["timestamp_bins"]["bins"],
                ],
                user_max=self.config["patient_id_max"],
                should_tokenize=True,
                cache=self.config["audit_log_cache"],
                max_length=self.n_positions,
            )
            dset.load_from_log()

        # Cache the datasets in parallel
        # Load them sequentially after caching
        if self.debug:  # Allows for pdb usage regardless
            threads = 1
        elif self.reset_cache:  # Multithreaded if not debugging and resetting cache
            threads = -1
        else:
            threads = (
                1  # Empirically faster to load sequentially if not resetting cache
            )

        joblib.Parallel(n_jobs=threads, verbose=1)(
            joblib.delayed(log_load)(self, provider)
            for provider in os.listdir(data_path)
        )

    def setup(self, stage=None):
        # Load the datasets
        path_prefix = ""
        for prefix in self.config["path_prefix"]:
            if os.path.exists(prefix):
                path_prefix = prefix
                break

        data_path = os.path.normpath(
            os.path.join(path_prefix, self.config["audit_log_path"])
        )
        datasets = []
        for provider in os.listdir(data_path):
            # Check there's a cache file (some should not have this, see above)
            prov_path = os.path.normpath(os.path.join(data_path, provider))
            if not os.path.exists(
                os.path.normpath(
                    os.path.join(prov_path, self.config["audit_log_cache"])
                )
            ):
                continue

            dset = EHRAuditDataset(
                prov_path,
                session_sep_min=self.config["sep_min"],
                log_name=self.config["audit_log_file"],
                vocab=self.vocab,
                timestamp_spaces=[
                    self.config["timestamp_bins"]["spacing"],
                    self.config["timestamp_bins"]["min"],
                    self.config["timestamp_bins"]["max"],
                    self.config["timestamp_bins"]["bins"],
                ],
                user_max=self.config["patient_id_max"],
                should_tokenize=False,
                cache=self.config["audit_log_cache"],
                max_length=self.n_positions,
            )
            if len(dset) != 0:
                datasets.append(dset)
            # Should automatically load from cache.

        config = yaml.safe_load(open("config.yaml", "r"))
        torch.manual_seed(config["random_seed"])
        self.seed = config["random_seed"]

        # Assign the datasets into different arrays of datasets to be chained together.
        # Shuffling will be done after separation of providers.
        train_indices, val_indices, test_indices = random_split(
            range(len(datasets)),
            [
                self.config["train_split"],
                self.config["val_split"],
                1 - self.config["train_split"] - self.config["val_split"],
            ],
        )
        self.train_dataset = ConcatDataset([datasets[i] for i in train_indices])
        self.val_dataset = ConcatDataset([datasets[i] for i in val_indices])
        self.test_dataset = ConcatDataset([datasets[i] for i in test_indices])
        self.num_workers = 2 if not self.debug else 1  # os.cpu_count()
        print(f"Using {self.num_workers} workers for data loading.")
        print(
            f"Train size: {len(train_indices)}, val size: {len(val_indices)}, test size: {len(test_indices)}"
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            worker_init_fn=partial(worker_fn, seed=self.seed),
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=partial(collate_fn, n_positions=self.n_positions),
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            worker_init_fn=partial(worker_fn, seed=self.seed),
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=partial(collate_fn, n_positions=self.n_positions),
            shuffle=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            worker_init_fn=partial(worker_fn, seed=self.seed),
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=partial(collate_fn, n_positions=self.n_positions),
            shuffle=True,
        )
