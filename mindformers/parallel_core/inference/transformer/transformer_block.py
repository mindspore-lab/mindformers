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
"""Transformer Block"""
__all__ = [
    'TransformerBlockSubmodules',
    'TransformerBlock'
]

from collections import OrderedDict
from dataclasses import dataclass
from typing import Union, List, Optional
from mindspore import nn, Tensor

from mindformers.parallel_core.inference.quantization import QuantizationConfig
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.inference.transformer.transformer_layer import BaseTransformerLayer
from mindformers.parallel_core.inference.transformer.norm import get_norm_cls
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs
from mindformers.parallel_core.inference.utils import get_num_layers_and_offset


@dataclass
class TransformerBlockSubmodules:
    """
    Class for specifying the submodules of a transformer block.

    This class defines the structure for configuring the layers and normalization
    within a transformer block, allowing for flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the transformer block. Each specification typically
            defines a complete transformer layer (e.g., self-attention, feed-forward network).
        layer_norm (Optional[Union[ModuleSpec, mindspore.nn.Cell]], optional): Specification
            or instance of the layer normalization to be applied.
    """

    layer_specs: List[ModuleSpec] = None
    layer_norm: Optional[Union[ModuleSpec, nn.Cell]] = None


def _get_block_submodules(
        config: TransformerConfig, spec: Union[TransformerBlockSubmodules, ModuleSpec]
) -> TransformerBlockSubmodules:
    """
    Retrieve or construct TransformerBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.

    Returns:
        TransformerBlockSubmodules: The submodules for the transformer block.
    """

    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    if isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        if issubclass(spec.module, BaseTransformerLayer):
            num_layers, _ = get_num_layers_and_offset(config)
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers, layer_norm=get_norm_cls(config.normalization)
            )
        raise Exception(f"specialize for {spec.module.__name__}.")
    raise Exception(f"specialize for {type(spec).__name__}.")


class TransformerBlock(nn.Cell):
    r"""
    Transformer class.

    Args:
        config (TransformerConfig): Configuration for transformer architecture.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Submodule specifications.
        post_layer_norm (bool): Whether to apply layer norm at the end of transformer block. Default: True.
        pre_process (bool): Default: False.
        post_process (bool): Default: False.
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.
        quant_config (QuantizationConfig, optional): Quantization configuration. Default: None.


    Inputs:
        - **hidden_states** (Tensor) - Input tensor.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **context (Tensor)** - Context tensor for cross-attention.
          Reserved parameter, is not currently supported. Default: None.
        - **context_mask** (Tensor) - Mask tensor for cross-attention.
          Reserved parameter, is not currently supported. Default: None.
        - **rotary_pos_cos** (Tensor) - The cos of rotary positional embeddings.
        - **rotary_pos_sin** (Tensor) - The sin of Rotary positional embeddings.
        - **attention_bias** (Tensor) - Bias tensor for Q * K.T.
          Reserved parameter, is not currently supported. Default: None.
        - **packed_seq_params** (Tensor) - Reserved parameter, is not currently supported. Default: None.
        - **batch_valid_length** (Tensor) - Tensor of shape [batch_size] for valid seq length.
        - **context_lens_tensor** (Tensor) - Tensor of context lengths.
        - **q_seq_lens** (Tensor) - Tensor of query lengths.
        - **block_tables** (Tensor) - Block tables for memory optimization.
        - **slot_mapping** (Tensor) - Slot mapping for memory optimization.
        - **attn_padding_idx** (Tensor) - Indices mapping positions in attention output sequence to
            original token positions, used for padding attention output to fixed size.
        - **attn_unpadding_idx** (Tensor) - Indices mapping valid tokens in padded attention output sequence to
            their original positions, used for removing padding in attention output.
        - **ffn_padding_idx** (Tensor) - Indices mapping positions in MoE output sequence to
            flattened valid token positions, used for padding MoE output to fixed size.
        - **ffn_unpadding_idx** (Tensor) - Indices mapping valid tokens in padded MoE output sequence to
            their original positions, used for removing padding in MoE output.
        - **key_cache** (Tensor, optional) - Key cache for incremental inference.
        - **value_cache** (Tensor, optional) - Value cache for incremental inference.

    Outputs:
        - **hidden_states** (Tensor) - Output tensor of transformer block

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            spec: Union[TransformerBlockSubmodules, ModuleSpec],
            post_layer_norm: bool = True,
            pre_process=False,
            post_process=False,
            model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super(TransformerBlock, self).__init__()
        self.config = config
        self.submodules = _get_block_submodules(config, spec)
        self.model_comm_pgs = model_comm_pgs
        self.post_layer_norm = post_layer_norm
        self.quant_config = quant_config
        self.pre_process = pre_process
        self.post_process = post_process
        self._build_layers(config)

    def _build_layers(self, config: TransformerConfig):
        '''The method to construct transformer layers'''
        # Transformer layers.
        def build_layer(layer_spec, layer_number):
            return build_module(layer_spec, config=self.config, layer_number=layer_number,
                                model_comm_pgs=self.model_comm_pgs, quant_config=self.quant_config,
                                prefix=f"decoder.layers.{layer_number-1}")
        self.num_layers, offset = get_num_layers_and_offset(config)
        layers_dict = OrderedDict()
        for i, layer_spec in enumerate(self.submodules.layer_specs):
            layers_dict[str(i + offset)] = build_layer(layer_spec, i + 1 + offset)
        self.layers = nn.SequentialCell(layers_dict)

        if self.submodules.layer_norm and self.post_layer_norm:
            self.final_layernorm = build_module(
                self.submodules.layer_norm,
                config=config,
                hidden_size=config.hidden_size,
                eps=config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def construct(
            self,
            hidden_states: Tensor,
            attention_mask: Tensor,
            context: Tensor = None,
            context_mask: Tensor = None,
            rotary_pos_cos: Tensor = None,
            rotary_pos_sin: Tensor = None,
            attention_bias: Tensor = None,
            packed_seq_params: Tensor = None,
            batch_valid_length=None,
            context_lens_tensor=None,
            q_seq_lens=None,
            block_tables=None,
            slot_mapping=None,
            attn_padding_idx=None,
            attn_unpadding_idx=None,
            ffn_padding_idx=None,
            ffn_unpadding_idx=None,
            key_cache=None,
            value_cache=None
    ):
        """ Construct function of transformer. """
        for index in range(self.num_layers):
            key_cache_idx = key_cache[index] if key_cache is not None else None
            value_cache_idx = value_cache[index] if value_cache is not None else None
            layer = self._get_layer(index)
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                batch_valid_length=batch_valid_length,
                context_lens_tensor=context_lens_tensor,
                q_seq_lens=q_seq_lens,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                attn_padding_idx=attn_padding_idx,
                attn_unpadding_idx=attn_unpadding_idx,
                ffn_padding_idx=ffn_padding_idx,
                ffn_unpadding_idx=ffn_unpadding_idx,
                key_cache=key_cache_idx,
                value_cache=value_cache_idx
            )

        # final layernorm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
