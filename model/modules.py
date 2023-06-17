import os

import torch
import yaml
from lightning import pytorch as pl
from transformers import PretrainedConfig

from Sophia.sophia import SophiaG
from model.data import EHRAuditDataset, EHRAuditTimestampBin, EHRAuditTokenize
from model.vocab import EHRVocab


class EHRAuditPretraining(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        self.step = 0

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch
        labels = (
            input_ids.clone().detach()
        )  # Should maybe use data collation in the future.
        outputs = self.model(input_ids, labels)
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
        input_ids = batch
        labels = input_ids.clone().detach()
        outputs = self.model(input_ids, labels)
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
        input_ids = batch
        labels = input_ids.clone().detach()
        outputs = self.model(input_ids, labels)
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

    def configure_optimizers(self):
        return SophiaG(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )


class EHRAuditDataModule(pl.LightningDataModule):
    def __init__(
        self,
        yaml_config_path: str,
    ):
        super().__init__()
        with open(yaml_config_path) as f:
            self.config = yaml.safe_load(f)

    def prepare_data(self):
        # Cannot set state here.
        pass

    def setup(self, stage=None):
        # Load the data from the audit log directory.
        data_path = self.config["audit_log_path"]
        log_name = self.config["audit_log_file"]
        sep_min = self.config["sep_min"]

        self.vocab = EHRVocab(vocab_path=self.config["vocab_path"])

        # Transforms
        transforms = [
            EHRAuditTimestampBin(
                timestamp_col="ACCESS_TIME",
                timestamp_spaces=[
                    self.config["timestamp_bins"]["min"],
                    self.config["timestamp_bins"]["max"],
                    self.config["timestamp_bins"]["bins"],
                ],
            ),
            EHRAuditTokenize(
                user_col="PAT_ID",
                timestamp_col="ACCESS_TIME",
                event_type_cols=["METRIC_NAME"],
                vocab=self.vocab,
            ),
        ]

        # Load the datasets
        datasets = []
        for provider in os.listdir(data_path):
            prov_path = os.path.join(data_path, provider)
            # Check the file is not empty and exists, there's a couple of these.
            log_path = os.path.join(prov_path, log_name)
            if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                continue
            dset = EHRAuditDataset(prov_path, sep_min=sep_min, log_name=log_name)
            for t in transforms:
                dset = t(dset)
            datasets.append(dset)

        self.datasets = datasets

        # Split the datasets
        train_size = int(len(self.datasets) * self.config["train_split"])
        val_size = int(len(self.datasets) * self.config["val_split"])
        test_size = len(self.datasets) - train_size - val_size
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(
            self.datasets, [train_size, val_size, test_size]
        )

        self.num_workers = os.cpu_count()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, num_workers=self.num_workers
        )
