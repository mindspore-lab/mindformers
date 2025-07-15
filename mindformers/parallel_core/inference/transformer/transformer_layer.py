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
"""Transformer Layer"""
__all__ = [
    'TransformerLayerSubmodules',
    'BaseTransformerLayer',
    'TransformerLayer'
]

from dataclasses import dataclass
from typing import Union, Optional

from mindspore import nn, ops

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.inference.transformer.identity_op import IdentityOp
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs


@dataclass
class TransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        post_self_attn_layernorm (Union[ModuleSpec, type]): Specification for the layer
            normalization after self-attention.
        pre_cross_attn_layernorm (Union[ModuleSpec, type]): Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        pre_mlp_layernorm (Union[ModuleSpec, type]): Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
        post_mlp_layernorm (Union[ModuleSpec, type]): Specification for the layer normalization
            after the MLP.
    """

    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    post_self_attn_layernorm: Union[ModuleSpec, type] = IdentityOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    post_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp


class BaseTransformerLayer:
    """A common parent class for `TransformerLayer` like implementations.

    A dummy class that is subclassed by similar `TransformerLayer`s e.g. the
    `TransformerLayer` in this file and possibly other `TransformerLayer`
    implementations that aim to use `TransformerBlock` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the `TransformerBlock`. See `_get_block_submodules` method
    implementation in `transformer_block.py` file for more details.
    """

    def __init__(self):
        pass


class TransformerLayer(nn.Cell, BaseTransformerLayer):
    r"""
    A single transformer layer.

    Args:
        config (TransformerConfig): Configuration for transformer architecture.
        submodules (TransformerLayerSubmodules): Submodule specifications.
        layer_number (int): Number of the transformer layer.
        hidden_dropout (float): The value of hidden dropout, do not need in inference. Default: None.
        model_comm_pgs (ModelCommProcessGroups, optional): Model communication process group.
            Default: default_model_comm_pgs.


    Inputs:
        - **hidden_states** (Tensor) - Input tensor.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **context (Tensor)** - Context tensor for cross-attention.
          Reserved parameter, is not currently supported. Default: None.
        - **context_mask (Tensor)** - Mask tensor for cross-attention.
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
        - **key_cache** (Tensor, optional) - Key cache for incremental inference.
        - **value_cache** (Tensor, optional) - Value cache for incremental inference.

    Outputs:
        - output (Tensor) - Output tensor of transformer layer

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self,
                 config: TransformerConfig,
                 submodules: TransformerLayerSubmodules,
                 layer_number: int = 1,
                 hidden_dropout: float = None,
                 model_comm_pgs: Optional[ModelCommProcessGroups] = default_model_comm_pgs,
                 ):
        super().__init__()
        if hidden_dropout and hidden_dropout != 0:
            # In inference mode, we want model output to be deterministic
            # and do not need to perform Dropout.
            raise ValueError(
                f"For TransformerLayer, `hidden_dropout` must be 0 or None, "
                f"but got `hidden_dropout={hidden_dropout}`"
            )

        self.config = config
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.apply_residual_connection_post_norm = config.apply_residual_connection_post_layernorm

        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        attention_optional_kwargs = {}
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            model_comm_pgs=model_comm_pgs,
            **attention_optional_kwargs,
        )

        self.post_self_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        # NOTE: pre_cross_attn_layernorm remains disabled here,
        # with GPTModel implementing it as Identity(X) initialization.
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        # NOTE: cross_attention remains disabled here,
        # with GPTModel implementing it as Identity(X) initialization.
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )

        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        self.mlp = build_module(submodules.mlp, config=self.config, model_comm_pgs=model_comm_pgs)

        self.post_mlp_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        self.add = ops.Add()

    def construct(
            self,
            hidden_states,
            attention_mask,
            context=None,
            context_mask=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            attention_bias=None,
            packed_seq_params=None,
            batch_valid_length=None,
            context_lens_tensor=None,
            q_seq_lens=None,
            block_tables=None,
            slot_mapping=None,
            key_cache=None,
            value_cache=None
    ):
        """
        Perform a forward pass through the transformer layer.

        This method implements the core computation of a transformer layer, including
        self-attention, cross-attention (is not supported currently), and feed-forward operations.
        """
        pre_mlp_layernorm_output, residual = self._construct_attention(
            hidden_states,
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
            key_cache=key_cache,
            value_cache=value_cache
        )
        output = self._construct_mlp(pre_mlp_layernorm_output, residual)
        return output

    def _construct_attention(
            self,
            hidden_states,
            attention_mask,
            context=None,
            context_mask=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            attention_bias=None,
            packed_seq_params=None,
            batch_valid_length=None,
            context_lens_tensor=None,
            q_seq_lens=None,
            block_tables=None,
            slot_mapping=None,
            key_cache=None,
            value_cache=None
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations (optional).

        Args:
            hidden_states (Tensor): Input tensor.
            attention_mask (Tensor): Tensor of attention mask.
            context (Tensor): Context tensor for cross-attention.
                Reserved parameter, is not currently supported. Default: None.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_cos (Tensor): The cos of rotary positional embeddings.
            rotary_pos_sin (Tensor): The sin of Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T.
                Reserved parameter, is not currently supported. Default: None.
            packed_seq_params (Tensor): Reserved parameter, is not currently supported. Default: None.
            batch_valid_length (Tensor): Tensor of shape [batch_size] for valid seq length.
            context_lens_tensor (Tensor): Tensor of context lengths.
            q_seq_lens (Tensor): Tensor of query lengths.
            block_tables (Tensor): Block tables for memory optimization.
            slot_mapping (Tensor): Slot mapping for memory optimization.
            key_cache (Tensor, optional): Key cache for incremental inference.
            value_cache (Tensor, optional): Value cache cache for incremental inference.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                pre_mlp_layernorm_output (Tensor): Transformed hidden states before the MLP.
                residual (Tensor): Residual connection.
        """
        # Layer norm at the beginning
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention
        attention_output = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            batch_valid_length=batch_valid_length,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            q_seq_lens=q_seq_lens,
            actual_seq_qlen=batch_valid_length,
            actual_seq_kvlen=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            key_cache=key_cache,
            value_cache=value_cache
        )

        # Optional Layer norm after self-attention
        attention_output = self.post_self_attn_layernorm(attention_output)

        # Residual connection
        if self.apply_residual_connection_post_norm:
            residual = input_layernorm_output
        else:
            residual = hidden_states

        residual = self.add(residual, attention_output)

        # Layer norm before MLP
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(residual)

        return pre_mlp_layernorm_output, residual

    def _construct_mlp(self, pre_mlp_layernorm_output, residual):
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            pre_mlp_layernorm_output (Tensor): Transformed hidden states before the MLP.
            residual (Tensor): Residual connection.

        Returns:
            output (Tensor): Transformed hidden states.
        """
        # MLP
        mlp_output = self.mlp(pre_mlp_layernorm_output)

        # Optional Layer norm after MLP
        mlp_output = self.post_mlp_layernorm(mlp_output)

        # Residual connection
        if self.apply_residual_connection_post_norm:
            residual = pre_mlp_layernorm_output
        else:
            residual = residual

        output = self.add(residual, mlp_output)
        return output
