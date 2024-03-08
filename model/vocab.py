# Used to build the vocab for the EHR audit log dataset.
import argparse
import itertools
from collections import OrderedDict
import os
import pickle
from typing import Dict, List

# Some portions Copyright 2020 IBM, licensed under Apache 2.0
# Some portions Copyright 2023 Hugging Face, licensed under Apache 2.0.
# Apache license header:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import yaml
import pandas as pd
from transformers import LogitsProcessor


class EHRVocab:
    def __init__(
        self,
        categorical_column_opts: Dict[str, List[str]] = None,
        vocab_path=None,
    ):
        """

        :param categorical_column_opts: Mapping of categorical column names to the list of possible values.
        :param max_len: Maximum length of the input sequence.
        :param vocab_path: Where to save/load the vocab.
        """
        if vocab_path is not None and os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f:
                self.__dict__.update(pickle.load(f))
        else:
            # Load the other vocab options from the config file.
            self.field_tokens = OrderedDict()
            self.field_ids = OrderedDict()
            self.global_tokens = OrderedDict()

            # Set the default vocab options for GPT-style models.
            self.special_tokens = {
                "eos_token": "<eos>",
                "unk_token": "<unk>",
            }

            def new_token(field, value):
                if field not in self.field_tokens:
                    self.field_tokens[field] = OrderedDict()
                    self.field_ids[field] = []

                self.field_tokens[field][value] = len(self.global_tokens)
                self.field_ids[field].append(len(self.global_tokens))
                self.global_tokens[len(self.global_tokens)] = (
                    field,
                    value,
                    len(self.field_ids[field]),
                )

            # Allocate the base tokens.
            for k, v in self.special_tokens.items():
                setattr(self, k, v)
                new_token("special", v)

            self.vocab_path = vocab_path

            # Allocate the categorical tokens.

            for category, tokens in categorical_column_opts.items():
                for t in tokens:
                    new_token(category, t)

    def save(self):
        with open(self.vocab_path, "wb") as f:
            res_dict = self.__dict__.copy()
            del res_dict["vocab_path"]
            pickle.dump(res_dict, f)

    def field_to_token(self, field, value):
        if value not in self.field_tokens[field]:
            # There's a few of these, TODO: Analyze these.
            return self.field_tokens["special"][self.unk_token]
        return self.field_tokens[field][value]

    def global_to_token(self, global_id):
        if global_id not in self.global_tokens:
            return 0
        return self.global_tokens[global_id] if global_id != 0 else (0, "special", 0)

    def globals_to_locals(self, global_ids: torch.Tensor):
        # Iterate over the elements of the tensor and convert them to local IDs.
        local_ids = torch.zeros_like(global_ids)
        global_ids_cpu = global_ids.cpu().numpy()
        for i in range(global_ids.shape[0]):  # Batch
            for j in range(global_ids.shape[1]):  # Element
                local_ids[i, j] = self.global_to_token(global_ids_cpu[i, j])

        return local_ids.to(global_ids.device)

    def globals_to_locals_torch(self, global_ids: torch.Tensor, field_start: int):
        # Idea: Each sub-vocab is always fixed in location, so the local ids are = global ids - offset.
        # If this assumption is not true, then it will break.
        return torch.clamp(torch.sub(global_ids, field_start - 1), min=0)

    def field_names(self, include_special=False):
        l = list(self.field_tokens.keys())
        if not include_special:
            l.remove("special")
        return l

    def __len__(self):
        return len(self.global_tokens)

# HuggingFace-style tokenizer implementing barebones tokenization for the EHR audit log dataset.
class EHRAuditTokenizer:
    def __init__(self, vocab: EHRVocab, timestamp_spaces_cal: List[float] = None,
                 user_col: str = "PAT_ID",
                 user_max: int = 128,
                 timestamp_col: str = "ACCESS_TIME",
                 timestamp_sort_cols: List[str] = ["ACCESS_TIME", "ACCESS_INSTANT"],
                 event_type_cols: List[str] = ["METRIC_NAME"],
                 max_length: int = 1024,
                 pat_ids_cat: bool = False):
        self.vocab = vocab
        self.timestamp_spaces_cal = timestamp_spaces_cal
        self.user_col = user_col
        self.user_max = user_max
        self.timestamp_col = timestamp_col
        self.timestamp_sort_cols = timestamp_sort_cols
        self.event_type_cols = event_type_cols
        self.max_length = max_length # Unused here
        self.pat_ids_cat = pat_ids_cat

    def encode(self, df: pd.DataFrame):
        """
        Encodes a an audit log DF into tokens.

        :param df:
        :return:
        """
        # Convert the timestamp to time deltas.
        # If not in seconds, convert to seconds.
        if df[self.timestamp_col].dtype == np.dtype("O"):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df[self.timestamp_col] = df[self.timestamp_col].astype(np.int64) // 10 ** 9

        if self.timestamp_sort_cols:
            df = df.sort_values(by=list(set(self.timestamp_sort_cols).intersection(df.columns)))

        # Time deltas, (ignore negative values, these will be quantized away)
        df.loc[:, self.timestamp_col] = df.loc[:, self.timestamp_col].copy().diff()

        # Set beginning of shift to 0, otherwise it's nan.
        df.loc[0, self.timestamp_col] = 0

        if self.pat_ids_cat:
            df[self.user_col] = df[self.user_col].astype("category").cat.codes

        # TODO: Ensure that the vocab responds to timestamp_bins.spacing
        if self.timestamp_spaces_cal is not None:
            res = df.loc[:, self.timestamp_col].apply(
                lambda x: np.digitize(
                    np.log(x + 1e-9), self.timestamp_spaces_cal
                ).astype(int)
            )
            # Drop the old timestamp column
            df = df.drop(columns=[self.timestamp_col])
            # Add the new timestamp column
            df[self.timestamp_col] = res.astype(int)

        tokenized_cols = self.event_type_cols + [self.user_col, self.timestamp_col]
        tokenized_exmaple = []
        for _, row in df.iterrows():
            tokenized_exmaple.extend(
                [
                    self.vocab.field_to_token(col, str(row[col]))
                    for col in tokenized_cols
                ]
            )
        return torch.Tensor(tokenized_exmaple)


    def decode(self,
               token_ids: List[int],
               output_type: type = None):
        """
        Decodes a list of token IDs into a list of field-value pairs and converts to the desired output type
        :param token_ids: List of token IDs to decode.
        :param output_type: Output type to convert to. If None, defaults to a Pandas DataFrame.
        :return: A list of field-value pairs.
        """
        fn = len(self.vocab.field_names(include_special=False))
        if output_type is None:
            output_type = pd.DataFrame

        rows = []
        row_dict = dict()

        for i in range(len(token_ids)):
            field, value, field_id = self.vocab.global_to_token(token_ids[i])
            if field == "ACCESS_TIME" and self.timestamp_spaces_cal is not None:
                # Dequantize the access time. field_id starts at 1.
                row_dict[field] = self.timestamp_spaces_cal[field_id - 1] if field_id != 1 else "<= 1"
            else:
                row_dict[field] = value
            if len(row_dict) == fn:
                rows.append(row_dict)
                row_dict = dict()
        return output_type(rows)

    def batch_decode(self,
                     token_ids: List[List[int]],
                     output_type: type = None):
        """
        Decodes a list of lists of token IDs into a list of lists of field-value pairs and converts to the desired output type
        :param token_ids: List of lists of token IDs to decode.
        :param output_type: Output type to convert to. If None, defaults to a Pandas DataFrame.
        :return: A list of lists of field-value pairs.
        """
        if output_type is None:
            output_type = pd.DataFrame

        return [self.decode(b, output_type=output_type) for b in token_ids]

class EHRAuditLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab: EHRVocab):
        self.vocab = vocab
        self.fields = self.vocab.field_names(include_special=False)
        self.fn = len(self.vocab.field_names(include_special=False))


    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor):
        # For each equivalent row in the vocab, we want to -inf out the logits that are outside the field range.
        # Assumes that each field in the batch is aligned.
        f = self.vocab.global_tokens[input_ids[:, -1].item()][0]
        next_field = (self.fields.index(f) + 1) % self.fn
        next_field_start = self.vocab.field_ids[self.fields[next_field]][0]
        next_field_end = self.vocab.field_ids[self.fields[next_field]][-1]
        logits[:, :next_field_start] = float("-inf")
        logits[:, next_field_end:] = float("-inf")
        return logits



if __name__ == "__main__":
    # Load the config
    with open(os.path.normpath(os.path.join(os.path.pardir, "config.yaml"))) as f:
        config = yaml.safe_load(f)

    # Determine the path prefix
    path_prefix = ""
    for prefix in config["path_prefix"]:
        if os.path.exists(prefix):
            path_prefix = prefix
            break

    # Erase the old vocab
    vocab_path = os.path.normpath(os.path.join(path_prefix, config["vocab_path"]))
    if os.path.exists(vocab_path):
        os.remove(vocab_path)

    # This is where we'll actually build the vocab and then save it.
    categorical_column_opts = dict()

    # METRIC_NAME
    df = pd.read_excel(
        os.path.normpath(os.path.join(path_prefix, config["metric_name_dict"]["file"])),
        engine="openpyxl",
    )
    categorical_column_opts["METRIC_NAME"] = df[
        config["metric_name_dict"]["column"]
    ].tolist()

    # Missing METRIC_NAMEs
    df2 = pd.read_excel(
        os.path.normpath(
            os.path.join(path_prefix, config["metric_name_dict"]["missing_metric_file"])
        ),
        engine="openpyxl",
    )
    categorical_column_opts["METRIC_NAME"].extend(
        # Extend with the column at index 0
        df2.iloc[:, 0].tolist()
    )

    # Build the patient IDs
    categorical_column_opts["PAT_ID"] = [
        # -1 = NaN
        str(i)
        for i in range(-1, config["patient_id_max"])
    ]

    # Time deltas
    bins = getattr(np, config["timestamp_bins"]["spacing"])(
        config["timestamp_bins"]["min"],
        config["timestamp_bins"]["max"],
        config["timestamp_bins"]["bins"],
    )

    categorical_column_opts["ACCESS_TIME"] = [str(i) for i in range(len(bins))]

    # Create the vocab
    vocab = EHRVocab(categorical_column_opts, vocab_path=vocab_path)
    vocab.save()

    # Print the vocab
    print("Field tokens:")
    print(vocab.field_tokens)
    print("Field IDs:")
    print(vocab.field_ids)
    print("Global tokens:")
    print(vocab.global_tokens)
