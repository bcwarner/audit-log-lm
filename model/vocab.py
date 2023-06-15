# Used to build the vocab for the EHR audit log dataset.
from collections import OrderedDict
import os
import pickle
from typing import Dict, List


class EHRVocab:
    def __init__(
        self,
        categorical_column_opts: Dict[str, List[str]],
        max_len=512,
        vocab_path=None,
    ):
        """

        :param categorical_column_opts: Mapping of categorical column names to the list of possible values.
        :param max_len: Maximum length of the input sequence.
        :param vocab_path: Where to save/load the vocab.
        """
        if vocab_path is not None:
            with open(vocab_path, "rb") as f:
                self.__dict__.update(pickle.load(f))
        else:
            # Load the other vocab options from the config file.
            self.field_tokens = OrderedDict()
            self.field_ids = OrderedDict()
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

            def new_token(field, value):
                self.field_tokens[field][value] = len(self.global_tokens)
                self.field_ids[field].append(len(self.global_tokens))
                self.global_tokens[len(self.global_tokens)] = (field, value)

            # Allocate the base tokens.
            for k, v in self.special_tokens.items():
                setattr(self, k, v)
                new_token("special", v)

            self.max_len = max_len
            self.vocab_path = vocab_path

            # Allocate the categorical tokens.
            for category, tokens in categorical_column_opts.items():
                for t in tokens:
                    new_token(category, t)

    def save(self):
        with open(self.vocab_path, "wb") as f:
            pickle.dump(self, f)

    def field_to_token(self, field, value):
        return self.field_tokens[field][value]

    def token_to_global(self, token):
        return self.global_tokens[token]

    def global_to_token(self, global_id):
        return self.global_tokens[global_id][1]

    def globals_to_locals(self, global_ids):
        return [self.global_to_token(g) for g in global_ids]

    def field_names(self):
        return list(self.field_tokens.keys())

    def __len__(self):
        return len(self.global_tokens)

    def __getitem__(self, key):
        return self.global_tokens[key]
