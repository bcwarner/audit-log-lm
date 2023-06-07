# Used to build the vocab for the EHR audit log dataset.
from collections import OrderedDict
import os
import pickle
from typing import Dict


class EHRVocab:
    def __init__(
        self, categorical_column_opts: Dict[str, str], max_len=512, vocab_path=None
    ):
        if vocab_path is not None:
            with open(vocab_path, "rb") as f:
                self.__dict__.update(pickle.load(f))
        else:
            # Load the other vocab options from the config file.
            self.field_tokens = OrderedDict()
            self.global_tokens = OrderedDict()

            # Set the default vocab options for GPT-style models.
            self.special_tokens = {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "eos_token": "<eos>",
                "bos_token": "<bos>",
                "mask_token": "<mask>",
                "sep_token": "<sep>",
                "cls_token": "<cls>",
            }
            for k, v in self.special_tokens.items():
                setattr(self, k, v)
                self.global_tokens[v] = [len(self.field_tokens), "special_tokens"]

            self.max_len = max_len

            self.vocab_path = vocab_path

            for category, tokens in categorical_column_opts.items():
                self.field_tokens[category] = OrderedDict(
                    {token: idx for idx, token in enumerate(tokens)}
                )
                for token in tokens:
                    self.global_tokens[token] = [len(self.global_tokens) - 1, category]

    def save(self):
        with open(self.vocab_path, "wb") as f:
            pickle.dump(self, f)

    def __len__(self):
        return len(self.global_tokens)

    def __getitem__(self, key):
        return self.global_tokens[key]
