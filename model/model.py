# Contains models for the EHR audit log dataset.
import os
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

# Dataloader for EHR audit log dataset, based on tabular transformers from Padhi et al. (2021)
# Portions adapted from HF transformers.
from typing import List, Optional, Union

import torch
from transformers import RwkvForCausalLM, RwkvConfig, TransfoXLLMHeadModel, PretrainedConfig, LlamaForCausalLM
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast
from transformers.models.rwkv.modeling_rwkv import RwkvCausalLMOutput

from model.vocab import EHRVocab


# Models worth trying:
# - GPT2
# - Longformer
# - Transformer-XL
# - RWKV

# May want to try using _WeightedLoss, may have optimization benefits.
class TabularLoss(torch.nn.Module):
    def __init__(self,
                 config: PretrainedConfig,
                 vocab: EHRVocab,
                 smoothing: List = None,
                 reduction: str = "mean"):
        super().__init__()
        self.seq_len = config.n_positions - 1
        self.vocab = vocab
        field_names = self.vocab.field_names(include_special=False)

        self.field_ct = len(field_names)
        self.loss = list(range(self.field_ct))
        self.col_ids = list(range(self.field_ct))
        self.col_ids_labels = list(range(self.field_ct))
        self.global_ids_min = list(range(self.field_ct))
        self.global_ids_max = list(range(self.field_ct))
        self.global_ids_len = list(range(self.field_ct))
        self.reduction = reduction
        if reduction not in ["mean", "none"]:
            raise NotImplementedError(f"Reduction {reduction} not implemented.")

        for field_idx, field_name in enumerate(field_names):
            self.loss[field_idx] = torch.nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction="none",  # Reduction is handled later since seq length might not be even.
                label_smoothing=smoothing[field_idx] if smoothing else 0.0,
            )
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

            self.global_ids_min[field_idx] = self.vocab.field_ids[field_name][0]
            self.global_ids_max[field_idx] = self.vocab.field_ids[field_name][-1] + 1
            self.global_ids_len[field_idx] = len(self.vocab.field_ids[field_name])

        # Tensorize the above fields.
        self.col_ids = torch.tensor(self.col_ids, dtype=torch.long)
        self.col_ids_labels = torch.tensor(self.col_ids_labels, dtype=torch.long)
        self.global_ids_min = torch.tensor(self.global_ids_min, dtype=torch.long)
        self.global_ids_max = torch.tensor(self.global_ids_max, dtype=torch.long)
        self.global_ids_len = torch.tensor(self.global_ids_len, dtype=torch.long)

    def forward(self, lm_logits, labels):
        # Shift so that tokens < n predict n
        # We don't need to remove the special labels here as they are not included here.
        shift_logits = lm_logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        # Shift so that tokens < n predict n
        # We don't need to remove the special labels here as they are not included here.
        shift_logits = lm_logits[..., :-1, :]  # .contiguous()
        shift_labels = labels[..., 1:]  # .contiguous()

        # Iterate through each of the fields and compute the loss over each column.
        def _compute_loss(field_idx):
            # Get the locations of the current column in the input.
            col_ids = self.col_ids[field_idx]

            # Get the locations of the current column in the labels.
            col_ids_labels = self.col_ids_labels[field_idx]

            # Get the IDs of the logits for the current column.
            global_ids_min = self.global_ids_min[field_idx]
            global_ids_max = self.global_ids_max[field_idx]
            global_ids_len = self.global_ids_len[field_idx]

            # Select the relevant logits.
            lm_logits_field = shift_logits[
                              :, col_ids, global_ids_min:global_ids_max
                              ]

            # Select the relevant labels.
            lm_labels_field = shift_labels[:, col_ids_labels]

            lm_labels_local_field_subbed = torch.clamp(
                torch.sub(lm_labels_field, global_ids_min), min=0
            )
            lm_labels_local_field = torch.where(
                lm_labels_field == -100, -100, lm_labels_local_field_subbed
            )

            # Compute the loss for the current column.
            lm_loss_field = self.loss[field_idx](
                lm_logits_field.transpose(2, 1),
                lm_labels_local_field,
            )
            return lm_loss_field

        # Doesn't get rid of the last dimension, but that's okay.
        losses = torch.cat([_compute_loss(field_idx) for field_idx in range(self.field_ct)])
        if self.reduction == "none":
            # Interleave the losses in order of column.
            total_lm_loss = losses
        elif self.reduction == "mean":
            # Take the mean across
            total_lm_loss = torch.mean(losses[losses > 0])
        else:
            raise NotImplementedError(f"Reduction {self.reduction} not implemented.")
        return total_lm_loss


class EHRAuditGPT2(GPT2LMHeadModel):
    def __init__(self, config, vocab: EHRVocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self.loss = TabularLoss(config, vocab, reduction="mean")

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
        should_break=False,
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

        total_lm_loss = None
        if labels is not None:
            # Compute the loss.
            total_lm_loss = self.loss(lm_logits, labels)

            # Append the loss to the end of the outputs.
        if not return_dict:
            outputs = (lm_logits,) + transformer_outputs[1:]
            return (total_lm_loss,) + outputs if total_lm_loss is not None else outputs

        return CausalLMOutputWithCrossAttentions(
            loss=total_lm_loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

class EHRAuditTransformerXL(TransfoXLLMHeadModel):
    """

    """
    def __init__(self, config, vocab: EHRVocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self.loss = TabularLoss(config, vocab, reduction="mean")

    def forward(
        self,
        input_ids=None,
        mems=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
        should_break=False,
        **kwargs
    ):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        total_lm_loss = None
        if labels is not None:
            # Compute the loss.
            total_lm_loss = self.loss(lm_logits, labels)

            # Append the loss to the end of the outputs.
        if not return_dict:
            outputs = (lm_logits,) + transformer_outputs[1:]
            return (total_lm_loss,) + outputs if total_lm_loss is not None else outputs

        return CausalLMOutputWithCrossAttentions(
            loss=total_lm_loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

class EHRAuditRWKV(RwkvForCausalLM):
    def __init__(self, config, vocab: EHRVocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self.loss = TabularLoss(config, vocab, reduction="mean")

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
        should_break=False,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        rwkv_outputs = self.rwkv(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = rwkv_outputs[0]

        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)

        if not return_dict:
            output = (logits,) + rwkv_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return RwkvCausalLMOutput(
            loss=loss,
            logits=logits,
            state=rwkv_outputs.state,
            hidden_states=rwkv_outputs.hidden_states,
            attentions=rwkv_outputs.attentions,
        )

class EHRAuditLlama(LlamaForCausalLM):
    def __init__(self, config, vocab: EHRVocab):
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self.loss = TabularLoss(config, vocab, reduction="mean")

    def forward(
            self,
            input_ids= None,
            attention_mask= None,
            position_ids= None,
            past_key_values= None,
            inputs_embeds= None,
            labels= None,
            use_cache = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
            should_break = False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )