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
"""Multi-Token Prediction (MTP) module for parallel token prediction.

This module enables the model to predict multiple future tokens in a single forward pass,
enhancing training efficiency and context awareness. It can be integrated with
speculative decoding for faster inference by generating draft tokens in parallel.

"""

__all__ = ['MultiTokenPredictionLayer', 'MultiTokenPredictionLayerSubmodules', 'get_mtp_layer_spec']

from dataclasses import dataclass
from typing import Optional, Union

from mindspore import nn, ops
from mindformers.parallel_core.inference.quantization import QuantizationConfig
from mindformers.parallel_core.inference.tensor_parallel.layers import ColumnParallelLinear
from mindformers.parallel_core.inference.transformer.norm import get_norm_cls
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module

def get_mtp_layer_spec(transformer_layer_spec: ModuleSpec, fused_norm=True) -> ModuleSpec:
    """Get the MTP layer spec.

    Returns:
        ModuleSpec: Module specification of MultiTokenPredictionLayer.
        fused_norm (bool): Whether to use fused-normalization. Defaults to True.
    """
    mtp_layer_spec = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=get_norm_cls(fused_norm),
            hnorm=get_norm_cls(fused_norm),
            eh_proj=ColumnParallelLinear,
            transformer_layer=transformer_layer_spec,
            layer_norm=get_norm_cls(fused_norm),
        ),
    )

    return mtp_layer_spec

@dataclass
class MultiTokenPredictionLayerSubmodules:
    """
    Dataclass for specifying the submodules of a MultiTokenPrediction module.

    Args:
        hnorm (Union[ModuleSpec, type]): Specification or instance of the
             hidden states normalization to be applied.
        enorm (Union[ModuleSpec, type]): Specification or instance of the
            embedding normalization to be applied.
        eh_proj (Union[ModuleSpec, type]): Specification or instance of the
            linear projection to be applied.
        transformer_layer (Union[ModuleSpec, type]): Specification
            or instance of the transformer block to be applied.
    """
    enorm: Union[ModuleSpec, type] = None
    hnorm: Union[ModuleSpec, type] = None
    eh_proj: Union[ModuleSpec, type] = None
    transformer_layer: Union[ModuleSpec, type] = None
    layer_norm: Union[ModuleSpec, type] = None


class MultiTokenPredictionLayer(nn.Cell):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    for more information, please refer to DeepSeek-V3 Technical Report
    https://arxiv.org/abs/2412.19437
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: MultiTokenPredictionLayerSubmodules,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.submodules = submodules
        self.dtype = config.compute_dtype

        self.enorm = build_module(
            self.submodules.enorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        self.hnorm = build_module(
            self.submodules.hnorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        self.eh_proj = build_module(
            self.submodules.eh_proj,
            config.hidden_size * 2,
            config.hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
        )

        self.transformer = build_module(self.submodules.transformer_layer, config=config,
                                        model_comm_pgs=model_comm_pgs, quant_config=quant_config)

        self.final_layernorm = build_module(
            self.submodules.layer_norm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        self.concat = ops.Concat(axis=-1)
        self.concat_mp = ops.Concat(axis=-1)
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()


    def construct(self, decoder_input, hidden_states, attention_mask=None, rotary_pos_cos=None,
                  rotary_pos_sin=None, batch_valid_length=None, context_lens_tensor=None, q_seq_lens=None,
                  block_tables=None, slot_mapping=None, attn_padding_idx=None, attn_unpadding_idx=None,
                  ffn_padding_idx=None, ffn_unpadding_idx=None, key_cache=None, value_cache=None):
        """ Perform the forward pass through the MTP layer. """

        key_cache = key_cache[0] if key_cache is not None else None
        value_cache = value_cache[0] if value_cache is not None else None

        decoder_input = self.enorm(decoder_input)
        hidden_states = self.hnorm(hidden_states)

        # At the (k - 1)-th MTP module, concatenates the i-th tocken's hidden_states
        # and the (i + K)-th tocken's embedding, and combine them with linear projection.
        hidden_states = self.concat((decoder_input, hidden_states))
        hidden_states = self.eh_proj(hidden_states)
        hidden_states = self.transformer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            q_seq_lens=q_seq_lens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            attn_padding_idx=attn_padding_idx,
            attn_unpadding_idx=attn_unpadding_idx,
            ffn_padding_idx=ffn_padding_idx,
            ffn_unpadding_idx=ffn_unpadding_idx,
            key_cache=key_cache,
            value_cache=value_cache
        )
        output = self.final_layernorm(hidden_states)
        return output
