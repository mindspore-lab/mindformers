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
"""Qwen2 models' APIs."""

from typing import Dict

import mindspore.common.dtype as mstype
from mindspore import Tensor, ops
from mindspore.communication import get_group_size
from mindspore.communication._comm_helper import _is_initialized

from mindformers.experimental.parallel_core.pynative.parallel_state import (get_group_info,
                                                                            initialize_model_parallel)
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.logger import logger
from mindformers.experimental.infer.core.utils import get_tp_world_size
from mindformers.experimental.infer.transformer.norm import get_norm
from mindformers.experimental.infer.transformer.mlp import MLP, MLPSubmodules
from mindformers.experimental.infer.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.experimental.graph.transformer.spec_utils import ModuleSpec
from mindformers.experimental.graph.transformer.transformer_config_utils import convert_to_transformer_config
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.infer.transformer.self_attention import (
    CoreAttention,
    SelfAttention,
    SelfAttentionSubmodules,
)
from mindformers.experimental.infer.transformer.flash_attention import FlashAttention
from mindformers.experimental.infer.core.gpt_model import GPTModel
from mindformers.experimental.models.qwen2.utils import Qwen2PreTrainedModel

__all__ = ["InferenceQwen2ForCausalLM"]


def get_gpt_layer_spec(config) -> ModuleSpec:
    r"""
    build gpt layer.

    Args:
        config (Qwen2Config): The config of qwen2 model.

    Returns:
        ModuleSpec: gpt layer

    """
    from mindformers.experimental.infer.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
    self_attn = ModuleSpec(
        module=SelfAttention,
        submodules=SelfAttentionSubmodules(
            core_attention=FlashAttention if config.use_flash_attention else CoreAttention,
            linear_proj=RowParallelLinear,
            linear_qkv=ColumnParallelLinear if config.qkv_concat else None,
            linear_q=ColumnParallelLinear if not config.qkv_concat else None,
            linear_k=ColumnParallelLinear if not config.qkv_concat else None,
            linear_v=ColumnParallelLinear if not config.qkv_concat else None
        )
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=get_norm(config),
            self_attention=self_attn,
            pre_mlp_layernorm=get_norm(config),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear
                )
            )
        )
    )


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class InferenceQwen2ForCausalLM(Qwen2PreTrainedModel):
    r"""
    Provide qwen2 model infer through network.

    Args:
        config (Qwen2Config): The config of qwen2 model.

    Returns:
        output: Tensor, the output of qwen2 decoderlayer

    """

    def __init__(self, config):
        super().__init__(config, auto_prefix=False)
        if get_group_info('tp').group is None and _is_initialized():
            initialize_model_parallel(get_group_size(), order='tp')
        transformer_config = TransformerConfig()
        self.config = convert_to_transformer_config(config, transformer_config)
        self.pad_token_id = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size
        self.max_position_embeddings = self.config.max_position_embeddings
        self.compute_dtype = self.config.compute_dtype

        self.cast = ops.Cast()
        self.gather = ops.Gather()
        self.sub = ops.Sub()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.tp_group_size = get_tp_world_size()
        self.is_prefill = True
        if isinstance(self.config.parallel_decoding_params, Dict):
            self.plugin_type = self.config.parallel_decoding_params.get("plugin_type")
        else:
            self.plugin_type = None
        self.model = GPTModel(config=self.config,
                              transformer_layer_spec=get_gpt_layer_spec(self.config),
                              vocab_size=self.vocab_size,
                              rotary_base=self.config.rotary_base)

    def set_dynamic_inputs(self, **kwargs):
        """ dynamic shape"""
        dynamic_input_ids = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_positions = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_context_lens_tensor = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_q_seq_lens = Tensor(shape=[None], dtype=mstype.int32)
        if self.plugin_type == "la":
            dynamic_attention_mask = Tensor(shape=[None, None], dtype=self.compute_dtype)
        else:
            dynamic_attention_mask = None
        self.set_inputs(dynamic_input_ids, dynamic_positions, dynamic_batch_valid_length,
                        dynamic_context_lens_tensor, dynamic_q_seq_lens, dynamic_block_tables,
                        dynamic_slot_mapping, None, dynamic_attention_mask, None)
        logger.info("Set dynamic input for qwen2.")

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
            layer.self_attention.flash_attention.add_flags(is_prefill=is_prefill)

    # pylint: disable=W0613
    def construct(self, input_ids, positions=None, batch_valid_length=None, context_lens_tensor=None, q_seq_lens=None,
                  block_tables=None, slot_mapping=None, kv_cache=None, attention_mask=None, attn_metadata=None):
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
            attn_metadata: attention metadata

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
            kv_cache=kv_cache,
            attn_metadata=attn_metadata
        )
        return logits
