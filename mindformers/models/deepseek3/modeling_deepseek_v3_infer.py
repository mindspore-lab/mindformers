# Copyright 2025 Huawei Technologies Co., Ltd
#
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
# ============================================================================
"""Deepseek-V3 models' APIs."""
from typing import Dict

from mindspore.communication import get_group_size
from mindspore.communication._comm_helper import _is_initialized

from mindformers.models.utils import jit
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config
from mindformers.models.deepseek3.utils import DeepseekV3PreTrainedModel
from mindformers.parallel_core.inference.parallel_state import (
    get_group_info,
    initialize_model_parallel
)
from mindformers.parallel_core.inference.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.inference.base_models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from mindformers.parallel_core.inference.model_utils import InferModelMixin

from .configuration_deepseek_v3 import DeepseekV3Config


class InferenceDeepseekV3ForCausalLM(DeepseekV3PreTrainedModel, InferModelMixin):
    r"""
    Provide Deepseek3 model infer through network.

    Args:
        config (Deepseek3Config): The config of deepseek3 model.

    Returns:
        output: Tensor, the output of qwen3 deepseek3 layer

    """

    def __init__(self, config: DeepseekV3Config):
        super().__init__(config, auto_prefix=False)
        if get_group_info('tp').group is None and _is_initialized():
            initialize_model_parallel(get_group_size(), order='tp')
        self.config = config
        config = convert_to_transformer_config(
            self.config,
            is_mla_model=True,
        )
        self.pad_token_id = self.config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.compute_dtype = config.compute_dtype

        self.is_prefill = True
        if isinstance(self.config.parallel_decoding_params, Dict):
            self.plugin_type = self.config.parallel_decoding_params.get("plugin_type")
        else:
            self.plugin_type = None
        self.model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_decoder_block_spec(
                config=config,
                normalization=config.normalization,
                qk_l2_norm=False,
            ),
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_position_embeddings,
            position_embedding_type=config.position_embedding_type,
            rotary_base=config.rotary_base
        )

    def add_flags_custom_mcore(self, is_prefill):
        r"""
        Add flag to distinguish fa and pa.

        Args:
            is_prefill: flag to distinguish fa and pa.

        Returns:

        """
        self.add_flags(is_prefill=is_prefill)
        self.model.add_flags(is_prefill=is_prefill)
        self.model.decoder.add_flags(is_prefill=is_prefill)
        self.model.casual_mask.add_flags(is_prefill=is_prefill)
        for layer in self.model.decoder.layers:
            layer.self_attention.add_flags(is_prefill=is_prefill)
            layer.self_attention.core_attention.add_flags(is_prefill=is_prefill)

    @jit
    def construct(
            self,
            input_ids,
            positions=None,
            batch_valid_length=None,
            context_lens_tensor=None,
            q_seq_lens=None,
            block_tables=None,
            slot_mapping=None,
            attention_mask=None,
            attn_metadata=None,
            key_cache=None,
            value_cache=None
    ):
        r"""
        model forward.

        Args:
            input_ids: input ids.
            positions: position ids.
            batch_valid_length: actual seq length.
            context_lens_tensor: computed key value length.
            block_tables: Store mapping tables for each sequence.
            slot_mapping : Token cache physical slot index.
            kv_cache: key cache and value cache.
            attention_mask: attentino mask used for fa or pa.
            attn_metadata: attention metadata.
            key_cache: key cache for incremental inference.
            value_cache: value cache for incremental inference.

        Returns:
            logits: the output logits.

        """
        logits = self.model(
            input_ids=input_ids,
            positions=positions,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            q_seq_lens=q_seq_lens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            attention_mask=attention_mask,
            attn_metadata=attn_metadata,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        return logits
