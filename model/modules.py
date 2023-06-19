import os

import torch
import yaml
from lightning import pytorch as pl
from tqdm import tqdm
from transformers import PretrainedConfig

from Sophia.sophia import SophiaG
from model.data import EHRAuditDataset
from model.vocab import EHRVocab


class EHRAuditPretraining(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        self.step = 0

    def token_prep(self, batch):
        # Returns input and labels
        # Should maybe use data collation in the future.
        input_ids = batch
        # Pad to max length or crop to max length
        if input_ids.size(1) < self.model.config.n_positions:
            input_ids = torch.nn.functional.pad(
                input=input_ids,
                pad=(0, self.model.config.n_positions - input_ids.shape[1]),
                value=0,
            )
        elif input_ids.size(1) > self.model.config.n_positions:
            input_ids = input_ids[:, : self.model.config.n_positions]

        labels = input_ids.clone().detach()
        return input_ids, labels

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids, labels = self.token_prep(batch)
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
        input_ids, labels = self.token_prep(batch)
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
        input_ids, labels = self.token_prep(batch)
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

        # Load the datasets
        data = []
        for provider in tqdm(os.listdir(data_path)):
            prov_path = os.path.join(data_path, provider)
            # Check the file is not empty and exists, there's a couple of these.
            log_path = os.path.join(prov_path, log_name)
            if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                continue
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
            )
            data.extend(dset.seqs)
            break

        self.data = data

        # Split the datasets
        train_size = int(len(self.data) * self.config["train_split"])
        val_size = int(len(self.data) * self.config["val_split"])
        test_size = len(self.data) - train_size - val_size
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(self.data, [train_size, val_size, test_size])

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
