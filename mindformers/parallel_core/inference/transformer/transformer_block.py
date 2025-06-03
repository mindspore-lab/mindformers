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

from dataclasses import dataclass
from typing import Union, List, Optional
from mindspore import nn, Tensor

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.inference.transformer.transformer_layer import BaseTransformerLayer
from mindformers.parallel_core.inference.transformer.norm import get_norm_cls


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
            num_layers = config.num_layers
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
        - **kv_cache** (List[Tensor], optional) - Key-value cache for incremental inference.

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
    ):
        super(TransformerBlock, self).__init__()

        # pre_process=True is not supported in TransformerBlock.
        # The corresponding Megatron v0.12.0 module's forward pass has this logic disabled by default,
        # so it won't cause significant impact.
        if pre_process:
            raise NotImplementedError("For TransformerBlock, `pre_process=True` is not supported.")

        # The post_process parameter is currently unused.
        # Since post_process is bound to post_layer_norm, it is executed based on post_layer_norm as a substitute.
        # The code behavior remains consistent with the corresponding module in Megatron v0.12.0.
        if post_process:
            raise NotImplementedError("For TransformerBlock, `post_process=True` is not supported.")

        self.config = config
        self.submodules = _get_block_submodules(config, spec)
        self.post_layer_norm = post_layer_norm
        self.num_layers = config.num_layers

        self._build_layers(config)

    def _build_layers(self, config: TransformerConfig):
        '''The method to construct transformer layers'''
        # Transformer layers.
        def build_layer(layer_spec, layer_number):
            return build_module(layer_spec, config=self.config, layer_number=layer_number)

        self.layers = nn.CellList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        if self.submodules.layer_norm and self.post_layer_norm:
            self.final_norm = build_module(
                self.submodules.layer_norm,
                config=config,
                hidden_size=config.hidden_size,
                eps=config.layernorm_epsilon,
            )
        else:
            self.final_norm = None

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
            kv_cache=None
    ):
        """ Construct function of transformer. """
        for index in range(self.num_layers):
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
                kv_cache=kv_cache
            )

        # final layernorm.
        if self.post_layer_norm:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states
