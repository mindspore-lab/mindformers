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

from mindformers.parallel_core.inference.quantization import QuantizationConfig
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.inference.transformer.mlp import MLP
from mindformers.parallel_core.inference.transformer.moe.moe_layer import MoELayer
from mindformers.parallel_core.inference.transformer.moe.experts import GroupedMLP
from mindformers.parallel_core.inference.transformer.identity_op import IdentityOp
from mindformers.parallel_core.inference.tensor_parallel.mappings import (gather_from_model_parallel_region,
                                                                          reduce_scatter_to_model_parallel_region,
                                                                          scatter_to_model_parallel_region)
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups, default_model_comm_pgs
from mindformers.tools.logger import logger


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
        quant_config (QuantizationConfig, optional): Quantization configuration. Default: None.
        prefix (str): The prefix string for this linear layer. Default: empty string("").


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
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
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
        self.moe_layer = False

        attention_optional_kwargs = {}
        additional_mlp_kwargs = {}

        if isinstance(submodules.mlp, ModuleSpec):
            mlp_module = submodules.mlp.module
            if mlp_module in (MoELayer, GroupedMLP):
                self.moe_layer = True
                # MoELayer expects model_comm_pgs to be passed in.
                additional_mlp_kwargs["model_comm_pgs"] = model_comm_pgs
            elif mlp_module == MLP:
                # MLP expects tp_group to be passed in.
                additional_mlp_kwargs["tp_group"] = model_comm_pgs.tp
            else:
                logger.warning(f"Current MLP module uses type: {type(submodules.mlp)}, "
                               f"which is not the type we recommend(MLP, GroupedMLP and MoELayer).")
        self.attn_delay_allreduce = not self.config.attn_allreduce and self.moe_layer
        self.attn_reduce_scatter = self.config.attn_reduce_scatter and self.moe_layer
        self.ffn_allgather = self.config.ffn_allgather and self.moe_layer
        self.need_padding = self.attn_reduce_scatter or (
            self.attn_delay_allreduce and not self.attn_reduce_scatter and not self.config.use_alltoall)

        self.tp_group = model_comm_pgs.tp

        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        attention_optional_kwargs["delay_allreduce"] = self.attn_delay_allreduce
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            model_comm_pgs=model_comm_pgs,
            **attention_optional_kwargs,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attention",
        )

        self.post_self_attn_layernorm = build_module(
            submodules.post_self_attn_layernorm,
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

        self.mlp = build_module(submodules.mlp, config=self.config,
                                quant_config=quant_config,
                                prefix=f"{prefix}.mlp", **additional_mlp_kwargs)

        self.post_mlp_layernorm = build_module(
            submodules.post_mlp_layernorm,
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
            attn_padding_idx=None,
            attn_unpadding_idx=None,
            ffn_padding_idx=None,
            ffn_unpadding_idx=None,
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
            attn_padding_idx=attn_padding_idx,
            key_cache=key_cache,
            value_cache=value_cache
        )
        output = self._construct_mlp(
            pre_mlp_layernorm_output,
            residual=residual,
            attn_unpadding_idx=attn_unpadding_idx,
            ffn_padding_idx=ffn_padding_idx,
            ffn_unpadding_idx=ffn_unpadding_idx
        )
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
            attn_padding_idx=None,
            key_cache=None,
            value_cache=None
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations (optional).
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

        if self.need_padding:
            attention_output = ops.gather(attention_output, attn_padding_idx, 0)
            hidden_states = ops.gather(hidden_states, attn_padding_idx, 0)
            if self.attn_reduce_scatter:
                attention_output = reduce_scatter_to_model_parallel_region(attention_output, self.tp_group)
                hidden_states = scatter_to_model_parallel_region(hidden_states, self.tp_group, dim=0)

        # Residual connection
        if self.apply_residual_connection_post_norm:
            residual = input_layernorm_output
        else:
            residual = hidden_states

        residual = self.add(residual, attention_output)

        # Layer norm before MLP
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(residual)

        return pre_mlp_layernorm_output, residual

    def _construct_mlp(
            self,
            pre_mlp_layernorm_output,
            residual,
            attn_unpadding_idx=None,
            ffn_padding_idx=None,
            ffn_unpadding_idx=None,
    ):
        """
        Perform a forward pass through the feed-forward layer.
        """
        # MLP
        mlp_extra_kwargs = {}
        if self.moe_layer:
            mlp_extra_kwargs = {
                "attn_unpadding_idx": attn_unpadding_idx,
                "ffn_padding_idx": ffn_padding_idx,
            }
        mlp_output = self.mlp(pre_mlp_layernorm_output, **mlp_extra_kwargs)

        # Optional Layer norm after MLP
        mlp_output = self.post_mlp_layernorm(mlp_output)

        # Residual connection
        if self.apply_residual_connection_post_norm:
            residual = pre_mlp_layernorm_output
        else:
            residual = residual

        output = self.add(residual, mlp_output)

        if self.ffn_allgather:
            output = gather_from_model_parallel_region(output, self.tp_group, dim=0)
        if self.attn_reduce_scatter:
            output = ops.gather(output, ffn_unpadding_idx, 0)
        return output
