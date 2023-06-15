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
        pass

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

    def __getitem__(self, item):
        if len(self.seqs) == 0:
            self.load_from_log()
        return self.seqs[item]

    def __len__(self):
        if len(self.seqs) == 0:
            self.load_from_log()
        return len(self.seqs)


class EHRAuditTimestampBin(object):
    """
    Set of transforms for the EHR audit log dataset.
    Assumes input is a set of shifts that has already had time deltas calculated and each user ID is a unique integer.

    Applies the following transforms:
        - Logarithmically scale the time deltas
    """

    def __init__(
        self,
        timestamp_col: str,
        timestamp_spaces: List[float],
        **kwargs,
    ):
        self.timestamp_col = timestamp_col
        self.timestamp_spaces = np.linspace(
            timestamp_spaces[0], timestamp_spaces[1], timestamp_spaces[2]
        )
        pass

    def __call__(self, seq):
        # Logarithmically scale the time deltas and then bin.
        # Based on (Padhi et al., 2021), but applies logarithmic binning.
        seq.seqs.loc[:, self.timestamp_col] = seq.seqs.loc[:, self.timestamp_col].apply(
            lambda x: np.digitize(np.log(x + 1), self.timestamp_spaces)
        )

        return seq


class EHRAuditTokenize(object):
    """
    Tokenizes the EHR audit log dataset with a built vocabulary.

        - Tokenize the event types
        - Tokenize the time deltas
        - Add the end of sequence token
        - Pad the sequences to the maximum length
        - Convert the sequences to tensors
    """

    def __init__(
        self,
        user_col: str,
        event_type_cols: List[str],
        timestamp_col: str,
        vocab: EHRVocab,
    ):
        self.user_col = user_col
        self.event_type_cols = event_type_cols
        self.timestamp_col = timestamp_col
        self.vocab = vocab
        pass

    def __call__(self, seq):
        # Convert each shift/session to tokenized sequences.
        tokenized_cols = [self.user_col, self.timestamp_col] + self.event_type_cols

        tokenized_example = []
        for i, row in seq.seqs.iterrows():
            tokenized_example.extend(
                [self.vocab.field_to_token(col, row[col]) for col in tokenized_cols]
            )

        # Add the end of sequence token.
        tokenized_example.append(self.vocab.special_tokens["eos_token"])

        # Pad the sequence to the maximum length.
        # TODO: Move this to the actual  model.
        max_len = self.vocab.max_len
        tokenized_example = tokenized_example[:max_len]
        tokenized_example = tokenized_example + [
            self.vocab.special_tokens["pad_token"]
        ] * (max_len - len(tokenized_example))

        # Convert the sequence to a tensor.
        tokenized_example = torch.tensor(tokenized_example)

        return tokenized_example
