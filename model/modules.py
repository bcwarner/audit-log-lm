import os
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


class EHRAuditPretraining(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        self.step = 0

    def forward(self, input_ids, labels):
        return self.model(input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(*batch)
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
    ):
        super().__init__()
        with open(yaml_config_path) as f:
            self.config = yaml.safe_load(f)
        self.vocab = vocab
        self.batch_size = batch_size
        self.n_positions = n_positions

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
        sep_min = self.config["sep_min"]

        def log_load(self, provider: str):
            print("Caching provider: ", provider)
            prov_path = os.path.normpath(os.path.join(data_path, provider))
            # Check the file is not empty and exists, there's a couple of these.
            log_path = os.path.normpath(os.path.join(prov_path, log_name))
            if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                return

            # Skip datasets we've already prepared.
            if os.path.exists(
                os.path.normpath(
                    os.path.join(prov_path, self.config["audit_log_cache"])
                )
            ):
                return

            dset = EHRAuditDataset(
                prov_path,
                sep_min=sep_min,
                log_name=log_name,
                vocab=self.vocab,
                timestamp_spaces=[
                    self.config["timestamp_bins"]["min"],
                    self.config["timestamp_bins"]["max"],
                    self.config["timestamp_bins"]["bins"],
                ],
                should_tokenize=True,
                cache=self.config["audit_log_cache"],
            )
            dset.load_from_log()

        # Cache the datasets in parallel
        # Load them sequentially after caching
        joblib.Parallel(n_jobs=1, verbose=1)(
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
        for provider in tqdm(os.listdir(data_path)):
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
                sep_min=self.config["sep_min"],
                log_name=self.config["audit_log_file"],
                vocab=self.vocab,
                timestamp_spaces=[
                    self.config["timestamp_bins"]["min"],
                    self.config["timestamp_bins"]["max"],
                    self.config["timestamp_bins"]["bins"],
                ],
                should_tokenize=False,
                cache=self.config["audit_log_cache"],
            )
            if len(dset) != 0:
                datasets.append(dset)
            # Should automatically load from cache.

        config = yaml.safe_load(open("config.yaml", "r"))
        torch.manual_seed(config["random_seed"])
        self.seed = config["random_seed"]

        # Assign the datasets into different arrays of datasets to be chained together.
        train_indices, val_indices, test_indices = random_split(
            range(len(datasets)),
            [
                self.config["train_split"],
                self.config["val_split"],
                1 - self.config["train_split"] - self.config["val_split"],
            ],
        )
        self.train_dataset = ChainDataset([datasets[i] for i in train_indices])
        self.val_dataset = ChainDataset([datasets[i] for i in val_indices])
        self.test_dataset = ChainDataset([datasets[i] for i in test_indices])
        self.num_workers = os.cpu_count()
        print(f"Using {self.num_workers} workers for data loading.")
        print(
            f"Train size: {len(train_indices)}, val size: {len(val_indices)}, test size: {len(test_indices)}"
        )

    def get_collate_fn(self):
        def collate_fn(batch):
            # Generate attention mask
            input_ids_col = []
            # attention_mask_col = []
            labels_col = []
            for input_ids in batch:
                pad_pos_count = self.n_positions - input_ids.size(0)
                # attention_mask = torch.ones_like(input_ids)
                # Pad to max length or crop to max length
                if input_ids.size(0) < self.n_positions:
                    # attention_mask = torch.nn.functional.pad(
                    #    input=attention_mask,
                    #    pad=(0, pad_pos_count),
                    #    value=0,  # Off for rest
                    # )
                    input_ids = torch.nn.functional.pad(
                        input=input_ids,
                        pad=(0, pad_pos_count),
                        value=0,  # EOS token
                    )
                elif input_ids.size(0) > self.n_positions:
                    input_ids = input_ids[: self.n_positions]
                    # attention_mask = attention_mask[:self.n_positions]

                labels = input_ids.clone().detach()

                input_ids_col.append(input_ids)
                # attention_mask_col.append(attention_mask)
                labels_col.append(labels)

            return torch.stack(input_ids_col), torch.stack(labels_col)

        return collate_fn

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            worker_init_fn=lambda _: torch.manual_seed(self.seed),
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=self.get_collate_fn(),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            worker_init_fn=lambda _: torch.manual_seed(self.seed),
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=self.get_collate_fn(),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            worker_init_fn=lambda _: torch.manual_seed(self.seed),
            pin_memory=True,
            batch_size=self.batch_size,
            collate_fn=self.get_collate_fn(),
        )
