# Contains models for the EHR audit log dataset.

import torch
from transformers import RwkvForCausalLM, RwkvConfig
from transformers import GPT2LMHeadModel, GPT2Config

# Models worth trying:
# - GPT2
# - Longformer
# - Transformer-XL
# - RWKV


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


class EHRAuditGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
