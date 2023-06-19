# Dataloader for EHR audit log dataset, based on vocabulary generation from Padhi et al. (2021)

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
import pandas as pd

from model.vocab import EHRVocab


class EHRAuditDataset(Dataset):
    """
    Dataset for Epic EHR audit log data.

    Assumes that the data is associated with a single physician.
    Users IDs are transformed to the # in which they are encountered for a given shift.
    Shifts are a gap in time set by hyperparameter, and have 0 entropy to start.
    Separation of shifts and sessions are delineated by the same process.
    Time deltas are calculated w.r.t. the preceding event.
    """

    def __init__(
        self,
        root_dir: str,
        sep_min: int = 240,
        user_col: str = "PAT_ID",
        timestamp_col: str = "ACCESS_TIME",
        event_type_cols: List[str] = ["METRIC_NAME"],
        log_name: str = None,
        vocab: EHRVocab = None,
        timestamp_spaces: List[float] = None,
        should_tokenize: bool = True,
    ):
        self.seqs = []
        self.provider = os.path.basename(root_dir)
        self.sep_min = sep_min
        self.user_col = user_col
        self.timestamp_col = timestamp_col
        self.event_type_cols = event_type_cols
        self.log_name = log_name
        self.root_dir = root_dir
        self.vocab = vocab
        self.timestamp_spaces = timestamp_spaces

        if self.timestamp_spaces is None and self.should_tokenize is True:
            raise ValueError("Tokenization depends on timestamp binning.")
        self.should_tokenize = should_tokenize

    def load_from_log(self):
        """
        Load the dataset from a log file.
        """
        path = os.path.join(self.root_dir, self.log_name)
        df = pd.read_csv(path)

        # Delete all columns not included
        df = df[[self.user_col, self.timestamp_col] + self.event_type_cols]

        # Convert the timestamp to time deltas.
        # If not in seconds, convert to seconds.
        if df[self.timestamp_col].dtype == np.dtype("O"):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df[self.timestamp_col] = df[self.timestamp_col].astype(np.int64) // 10**9
        df.loc[:, self.timestamp_col] = df.loc[:, self.timestamp_col].copy().diff()
        # Set beginning of shift to 0, otherwise it's nan.
        df.loc[0, self.timestamp_col] = 0

        # Separate the data into shifts.
        sep_sec = self.sep_min * 60
        seqs = []
        seq_start_idx = 0
        for i, row in df.iterrows():
            if row[self.timestamp_col] > sep_sec:
                df.loc[i, self.timestamp_col] = 0  # Reset the time delta to 0.
                seq_end_idx = i
                seqs.append(df.iloc[seq_start_idx:seq_end_idx, :].copy())
                seq_start_idx = seq_end_idx

        # Append the last shift
        seqs.append(df.iloc[seq_start_idx:, :].copy())

        # Convert the user IDs in each shift to a unique integer
        # Also convert the events to the corresponding vocab value.
        for seq in seqs:
            seq[self.user_col] = seq[self.user_col].astype("category").cat.codes
            # TODO: Decide on how to encode these best.

        self.seqs = seqs

        if self.timestamp_spaces is not None:
            for s in self.seqs:
                s.loc[:, self.timestamp_col] = s.loc[:, self.timestamp_col].apply(
                    lambda x: np.digitize(np.log(x + 1), self.timestamp_spaces)
                )

        if self.should_tokenize:
            tokenized_cols = [self.user_col, self.timestamp_col] + self.event_type_cols

            tokenized_seqs = []
            for s in self.seqs:  # Iterate each sequence
                tokenized_example = []
                for i, row in s.iterrows():  # Iterate each row
                    tokenized_example.extend(
                        [
                            self.vocab.field_to_token(col, str(row[col]))
                            for col in tokenized_cols
                        ]
                    )

                # Add the end of sequence token.
                tokenized_example.append(
                    self.vocab.field_to_token("special", self.vocab.eos_token)
                )

                # Convert the sequence to a tensor.
                tokenized_example = torch.tensor(tokenized_example)
                tokenized_seqs.append(tokenized_example)

            self.seqs = tokenized_seqs
            return

    def __getitem__(self, item):
        if len(self.seqs) == 0:
            self.load_from_log()
        return self.seqs[item]

    def __len__(self):
        if len(self.seqs) == 0:
            self.load_from_log()
        return len(self.seqs)
