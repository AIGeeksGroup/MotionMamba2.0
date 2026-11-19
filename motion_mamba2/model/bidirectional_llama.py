# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2024 McGill NLP
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# The above copyright notice and this permission notice shall be included in
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.
'Bidirectional llama.'
import torch
from peft import PeftModel
from torch import nn
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from transformers import LlamaModel
from transformers import LlamaPreTrainedModel
from transformers.cache_utils import Cache
from transformers.cache_utils import StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.utils import logging
from motion_mamba2.model.language_utils import is_transformers_attn_greater_or_equal_4_43_1
logger = logging.get_logger(__name__)

class ModifiedLlamaAttention(LlamaAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

class ModifiedLlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(self, config: LlamaConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = ModifiedLlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

class LlamaBiModel(LlamaModel):
    _no_split_modules = ['ModifiedLlamaDecoderLayer']

    def __init__(self, config: LlamaConfig):
        if not is_transformers_attn_greater_or_equal_4_43_1():
            raise ValueError('The current implementation of LlamaEncoderModel follows modeling_llama.py of transformers version >= 4.43.1')
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([ModifiedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values: Cache, output_attentions: bool):
        if self.config._attn_implementation == 'flash_attention_2':
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        (dtype, device) = (input_tensor.dtype, input_tensor.device)
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1
        causal_mask = torch.zeros((sequence_length, target_length), dtype=dtype, device=device)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = attention_mask.eq(0.0).to(dtype=dtype) * min_dtype
                causal_mask[:mask_shape[0], :mask_shape[1], offset:mask_shape[2] + offset, :mask_shape[3]] = mask_slice
        if self.config._attn_implementation == 'sdpa' and attention_mask is not None and (attention_mask.device.type == 'cuda') and (not output_attentions):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
        return causal_mask

class LlamaBiForMNTP(LlamaForCausalLM):

    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaBiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model_for_peft(self):
        return self.model

    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    def save_peft_model(self, path):
        self.model.save_pretrained(path)
