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
from dataclasses import dataclass
from typing import Union, List, Optional
from mindspore import nn, Tensor, dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.auto_generate import Reshape
from mindformers.parallel_core.training_graph.transformer.utils import LayerSetting
from mindformers.parallel_core.training_graph.transformer.norm import FusedNorm
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.training_graph.transformer.transformer_layer import BaseTransformerLayer
from mindformers.tools.logger import logger


@dataclass
class TransformerBlockSubmodules:
    """
    Class for specifying the submodules of a transformer block.

    This class defines the structure for configuring the layers and normalization
    within a transformer block, allowing for flexible and customizable architecture designs.

    Args:
        layer_specs (ModuleSpec, optional): A module specifications for
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
                layer_specs=[spec] * num_layers, layer_norm=FusedNorm
                # Only implements the FusedLayerNorm method for benchmarking purposes.
            )
        raise Exception(f"specialize for {spec.module.__name__}.")
    raise Exception(f"specialize for {type(spec).__name__}.")


class TransformerBlock(nn.Cell):
    r"""
    Transformer class.

    Args:
        config (dict): Configuration.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specs.
        post_layer_norm (bool): Insert normalization layer at the end of transformer block. Default: True.
        pre_process : Default None.
        post_process : Default None.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **rotary_pos_emb** (Tensor) - Tensor of rotary position embedding. Default: None.

    Outputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.

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

        self.submodules = _get_block_submodules(config, spec)
        self.post_layer_norm = post_layer_norm
        self.num_layers = config.num_layers
        cp = 1 if config is None else config.context_parallel_size
        self.compute_2d = (config.sequence_parallel and cp == 1)
        if config.sequence_parallel and cp > 1:
            logger.warning("The context paralley way conflicts with sequence with sequence parallel way."
                           "The sequence parallel way has no effect and ignored.")
        self.seq_length_in_cfg = config.seq_length

        # The CPU offloading implementation differs from Megatron v0.12.0's approach,
        # so no related scripts are implemented here.

        self._build_layers(config)

        self.shape = P.Shape()
        self.reshape_2d = Reshape()
        self.reshape_back = Reshape()
        self.init_extra_loss = Tensor([0], mstype.float32)

        self.shard(config)

    def _build_layers(self, config: TransformerConfig):
        """build transformer layers."""
        # Transformer layers.
        self.layers = nn.CellList()
        # layer setting, take mtp layers into total layers.
        mtp_num_layers = 0 if not config.mtp_num_layers else config.mtp_num_layers
        self.layer_setting = LayerSetting(config.num_layers + mtp_num_layers,
                                          config.offset,
                                          config,
                                          config.virtual_pipeline_model_parallel_size)
        for layer_id in range(config.num_layers):
            layer = build_module(self.submodules.layer_specs[layer_id], config=config, layer_number=layer_id)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)

        if self.post_layer_norm:
            self.final_norm = build_module(self.submodules.layer_norm,
                                           config=config,
                                           dim=config.hidden_size,
                                           eps=config.layernorm_epsilon,
                                           param_init_type=config.layernorm_compute_dtype)
        else:
            self.final_norm = None

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def construct(self,
                  hidden_states: Tensor,
                  attention_mask: Tensor,
                  rotary_pos_emb: Tensor = None,
                  prefix_keys_values=None,
                  actual_seq_len=None):
        """ Construct function of transformer. """
        seq_len, bs, hs = self.shape(hidden_states)
        if self.compute_2d:
            hidden_states = self.reshape_2d(hidden_states, (-1, hs))
            if seq_len != self.seq_length_in_cfg:
                raise ValueError("config.seq_length is not equal to sequence length of input!")

        extra_loss = self.init_extra_loss
        for index in range(self.num_layers):
            layer = self._get_layer(index)
            prefix_kv = prefix_keys_values[index] if prefix_keys_values is not None else None
            # pylint: disable=W0612
            hidden_states, context, extra_loss = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                prefix_keys_values=prefix_kv,
                extra_loss=extra_loss,
                actual_seq_len=actual_seq_len
                # context/context_mask/inference_context/packed_seq_params/sequence_len_offset is useless,
                # In Megatron v0.12.0, this is primarily used for inference-related processing and
                # has no practical impact on training.
                # attention_bias is useless by default, only used for T5 in Megatron v0.12.0.
            )

        # final layernorm.
        if self.post_layer_norm:
            hidden_states = self.final_norm(hidden_states)

        if self.compute_2d:
            hidden_states = self.reshape_back(hidden_states, (bs, seq_len, -1))

        return hidden_states, extra_loss

    def shard(self, config: TransformerConfig):
        """ shard function of mlp block. """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        if self.post_layer_norm:
            if self.compute_2d:
                self.final_norm.shard(config, in_strategy=(dp * cp, 1))
            else:
                self.final_norm.shard(config, in_strategy=(cp, dp, 1))
