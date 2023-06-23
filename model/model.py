# Contains models for the EHR audit log dataset.

import torch
from transformers import RwkvForCausalLM, RwkvConfig
from transformers import GPT2LMHeadModel, GPT2Config

from model.vocab import EHRVocab


# Models worth trying:
# - GPT2
# - Longformer
# - Transformer-XL
# - RWKV


class EHRAuditGPT2(GPT2LMHeadModel):
    def __init__(self, config, vocab: EHRVocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self.seq_len = config.n_positions - 1

        field_names = self.vocab.field_names(include_special=False)
        self.field_ct = len(field_names)
        self.col_ids = list(range(self.field_ct))
        self.col_ids_labels = list(range(self.field_ct))
        self.global_ids_field = list(range(self.field_ct))
        self.field_start = list(range(self.field_ct))
        for field_idx, field_name in enumerate(field_names):
            self.col_ids[field_idx] = list(
                range(field_idx, self.seq_len, len(field_names))
            )
            self.col_ids_labels[field_idx] = list(
                range(field_idx - 1, self.seq_len, len(field_names))
            )
            if field_idx == 0:
                self.col_ids_labels[field_idx] = self.col_ids_labels[field_idx][1:]

            if len(self.col_ids[field_idx]) < len(
                self.col_ids_labels[field_idx]
            ):  # Ensure the lengths are the same.
                self.col_ids[field_idx].append(
                    self.col_ids[field_idx][-1] + len(field_names)
                )

            if len(self.col_ids_labels[field_idx]) < len(self.col_ids[field_idx]):
                self.col_ids_labels[field_idx].append(
                    self.col_ids_labels[field_idx][-1] + len(field_names)
                )

            self.global_ids_field[field_idx] = self.vocab.field_ids[field_name]
            self.field_start[field_idx] = self.vocab.field_ids[field_name][0]

        # Tensorize the above fields.
        self.col_ids = torch.tensor(self.col_ids, dtype=torch.long)
        self.col_ids_labels = torch.tensor(self.col_ids_labels, dtype=torch.long)
        self.global_ids_field = [
            torch.tensor(x, dtype=torch.long) for x in self.global_ids_field
        ]
        self.field_start = torch.tensor(self.field_start, dtype=torch.long)

    def forward(
        self,
        input_ids=None,
        labels=None,
        attention_mask=None,
        past=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            # We don't need to remove the special labels here as they are not included here.
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            seq_len = shift_logits.size(1)
            # Ensure that the sequence len does not go past the attention mask for each batch
            total_lm_loss = 0

            # Iterate through each of the fields and compute the loss over each column.
            # Exclude the special tokens.
            # field_names = self.vocab.field_names(include_special=False)
            for field_idx in range(self.field_ct):
                # Get the locations of the current column in the input.
                col_ids = self.col_ids[field_idx]

                # Get the locations of the current column in the labels.
                col_ids_labels = self.col_ids_labels[field_idx]
                # if field_idx == 0:
                #    col_ids_labels = col_ids_labels[1:]

                # if len(col_ids) < len(
                #    col_ids_labels
                # ):  # Ensure the lengths are the same.
                #    col_ids.append(col_ids[-1] + len(field_names))

                # if len(col_ids_labels) < len(col_ids):
                #    col_ids_labels.append(col_ids_labels[-1] + len(field_names))

                # Get the IDs of the logits for the current column.
                global_ids_field = self.global_ids_field[field_idx]

                # Select the relevant logits.
                lm_logits_field = shift_logits[:, col_ids, :][:, :, global_ids_field]

                # Select the relevant labels.
                lm_labels_field = shift_labels[:, col_ids_labels]
                # breakpoint()
                lm_labels_local_field = self.vocab.globals_to_locals_torch(
                    lm_labels_field, self.field_start[field_idx]
                )

                # Compute the loss for the current column.
                loss_fct = torch.nn.CrossEntropyLoss()
                lm_loss_field = loss_fct(
                    lm_logits_field.view(-1, len(global_ids_field)),
                    lm_labels_local_field.view(-1),
                )
                total_lm_loss += lm_loss_field

            # Append the loss to the end of the outputs.
            outputs = (total_lm_loss,) + outputs

        return outputs


class EHRAuditRWKV(RwkvForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        state=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
