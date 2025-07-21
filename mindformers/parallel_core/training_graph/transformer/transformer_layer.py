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
from dataclasses import dataclass
from typing import Union
from mindspore.ops import auto_generate as aclnn_ops
from mindspore import nn
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.dropout import Dropout
from mindformers.parallel_core.training_graph.transformer.identity_op import IdentityOp


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
        pre_cross_attn_layernorm (Union[ModuleSpec, type]): Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        pre_mlp_layernorm (Union[ModuleSpec, type]): Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
    """

    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp


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
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self,
                 config: TransformerConfig,
                 submodules: TransformerLayerSubmodules,
                 layer_number: int = 1,
                 hidden_dropout: float = None,
                 ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.apply_residual_connection_post_norm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.use_eod_attn_mask_compression = config.use_eod_attn_mask_compression
        self.sequence_parallel = config.sequence_parallel

        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=config,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        attention_optional_kwargs = {}
        # NOTE: attention_optional_kwargs={}, Megatron v0.12.0 requires explicit communication setup here,
        # but MindSpore's built-in shard mechanism handles this automatically in Graph mode.
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )
        # self_attn_bda(BiasDropoutFusion) is not supported.
        # NOTE: JIT-Graph fusion optimizes performance at the cost of potential
        # computational throughput changes (precision remains unaffected).

        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=config,
            dim=config.hidden_size,
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

        # cross_attn_bda(BiasDropoutFusion) is not supported.
        # NOTE: JIT-Graph fusion optimizes performance at the cost of potential
        # computational throughput changes (precision remains unaffected).

        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=config,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        self.mlp = build_module(submodules.mlp, config=self.config)

        # mlp_bda(BiasDropoutFusion) is not supported.
        # NOTE: JIT-Graph fusion optimizes performance at the cost of potential
        # computational throughput changes (precision remains unaffected).

        self.hidden_states_dropout = Dropout(drop_prob=self.hidden_dropout)
        self.add = aclnn_ops.AddExt()
        self.add_bias = aclnn_ops.AddExt()

        # NOTE: Recompute configuration is managed by
        # training_graph.transformer.utils.LayerSetting
        # and applied in TransformerBlock. Check implementation for defaults.

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        else:
            self.shard(config)

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            prefix_keys_values=None,
            extra_loss=0.,
            actual_seq_len=None
    ):
        """
        Perform a forward pass through the transformer layer.

        This method implements the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                output (Tensor): Transformed hidden states of shape [s, b, h].
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """
        # context/context_mask are only used in cross_attention modules, unused in GPTModel.

        # Layer norm at the beginning
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Residual connection
        if self.apply_residual_connection_post_norm:
            residual = input_layernorm_output
        else:
            residual = hidden_states

        # Self-Attention
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            prefix_keys_values=prefix_keys_values,
            actual_seq_len=actual_seq_len
        )

        if isinstance(attention_output_with_bias, tuple):
            attention_output, self_attention_bias = attention_output_with_bias
        else:
            attention_output = attention_output_with_bias
            self_attention_bias = None

        if self_attention_bias is not None:
            attention_output = self.add_bias(attention_output, self_attention_bias)

        # Cross attention is Identity(X) in GPTModel, future expansions will be implemented.

        # Dropout
        dropout_output = self.hidden_states_dropout(attention_output)

        norm_input = self.add(residual, dropout_output)

        # Layer norm post the self attention
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(norm_input)

        # Residual connection
        if self.apply_residual_connection_post_norm:
            residual = pre_mlp_layernorm_output
        else:
            residual = norm_input

        mlp_output, mlp_output_bias, extra_loss = self.mlp(
            pre_mlp_layernorm_output,
            extra_loss=extra_loss
        )

        if mlp_output_bias is not None:
            mlp_output = self.add_bias(mlp_output, mlp_output_bias)

        # Dropout
        dropout_output = self.hidden_states_dropout(mlp_output)

        output = self.add(residual, dropout_output)
        # 'return context' is useless, this param may be deprecated later.
        return output, context, extra_loss

    def shard(self, config: TransformerConfig):
        """ shard function of mlp block. """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1

        if self.sequence_parallel or cp > 1:
            self.input_layernorm.shard(config, in_strategy=(cp * tp, dp, 1))
            self.pre_mlp_layernorm.shard(config, in_strategy=(cp * tp, dp, 1))
            self.add.shard(((cp * tp, dp, 1), (cp * tp, dp, 1)))
            self.hidden_states_dropout.shard((cp * tp, dp, 1))
            self.add_bias.shard(((cp * tp, dp, 1), (1,)))
        else:
            self.input_layernorm.shard(config, in_strategy=(1, dp, 1))
            self.pre_mlp_layernorm.shard(config, in_strategy=(1, dp, 1))
            self.add.shard(((1, dp, 1), (1, dp, 1)))
            self.hidden_states_dropout.shard((1, dp, 1))
            self.add_bias.shard(((1, dp, 1), (1,)))

    def sharding_propagation(self, config: TransformerConfig):
        pass
