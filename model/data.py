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
    We don't separate into sessions (separated by 5 or so minute gaps)
    Time deltas are calculated w.r.t. the preceding event.
    """

    def __init__(
        self,
        root_dir: str,
        shift_sep_hr: int = 4,
        user_cols: List[str] = ["PAT_ID"],
        timestamp_col: List[str] = "ACCESS_INSTANT",
        event_type_cols: List[str] = ["METRIC_NAME", "REPORT_NAME"],
        log_name: str = None,
        vocab: EHRVocab = None,
    ):
        self.examples = []
        self.shift_sep_hr = shift_sep_hr
        self.user_cols = user_cols
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
        df = df[self.user_cols + self.timestamp_col + self.event_type_cols]

        # Ensure that the dataframe is sorted by timestamp.
        df = df.sort_values(by=self.timestamp_col)

        # Convert the timestamp to time deltas.
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        df[self.timestamp_col] = df[self.timestamp_col].diff()
        df[self.timestamp_col] = df[self.timestamp_col].dt.total_seconds()

        # Separate the data into shifts.
        shift_sep_sec = self.shift_sep_hr * 60 * 60
        shifts = []
        shift_start_idx = 0
        for i, row in df.iterrows():
            if row[self.timestamp_col] > shift_sep_sec:
                df[self.timestamp_col][i] = 0  # Reset the time delta to 0.
                shift_end_idx = i
                shifts.append(df[shift_start_idx:shift_end_idx])
                shift_start_idx = shift_end_idx

        # Append the last shift
        shifts.append(df[shift_start_idx:])

        # Convert the user IDs in each shift to a unique integer
        # Also convert the events to the corresponding vocab value.
        for shift in shifts:
            for col in self.user_cols:
                shift[col] = shift[col].astype("category").cat.codes
                # TODO: Decide on how to encode these best.

    def __getitem__(self, item):
        if len(self.examples) == 0:
            self.load_from_log()
        return self.examples[item]

    def __len__(self):
        if len(self.examples) == 0:
            self.load_from_log()
        return len(self.examples)


class EHRAuditTransforms(object):
    """
    Set of transforms for the EHR audit log dataset.
    Assumes input is a set of shifts that has already had time deltas calculated and each user ID is a unique integer.

    Applies the following transforms:
        - Tokenize the event types
        - Logarithmically scale the time deltas
        - Tokenize the time deltas
        - Add the end of sequence token
        - Pad the sequences to the maximum length
        - Convert the sequences to tensors
    """

    def __init__(
        self,
        user_cols: List[str],
        event_type_cols: List[str],
        timestamp_col: str,
        vocab: EHRVocab,
    ):
        self.user_cols = user_cols
        self.event_type_cols = event_type_cols
        self.timestamp_col = timestamp_col
        self.vocab = vocab
        pass

    def __call__(self, shift):
        # Logarithmically scale the time deltas.
        # From (Padhi et al., 2021)
        shift[self.timestamp_col] = shift[self.timestamp_col].apply(
            lambda x: np.log(x + 1)
        )
        # TODO: Explore whether the REaLTabFormer strategy is better.

        # Convert each shift to tokenized sequences.
        tokenized_cols = self.user_cols + self.event_type_cols + [self.timestamp_col]

        tokenized_example = []
        for i, row in shift.iterrows():
            tokenized_example.extend(
                [self.vocab[row[col]][0] for col in tokenized_cols]
            )

        # Add the end of sequence token.
        tokenized_example.append(self.vocab.special_tokens["eos_token"])

        # Pad the sequence to the maximum length.
        max_len = self.vocab.max_len
        tokenized_example = tokenized_example[:max_len]
        tokenized_example = tokenized_example + [
            self.vocab.special_tokens["pad_token"]
        ] * (max_len - len(tokenized_example))

        # Convert the sequence to a tensor.
        tokenized_example = torch.tensor(tokenized_example)

        return tokenized_example
