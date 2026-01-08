# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Modified to adapt to MindSpore pynative mode.
# Main changes: PyTorch->MindSpore, removed parallel/checkpointing/CUDA graph/CPU offloading features,
# removed pre_process/post_process parameters, simplified forward pass.
"""Transformer Block"""
from dataclasses import dataclass
from typing import Union, List, Optional
from mindspore import nn, Tensor, dtype as mstype
from mindformers.pynative.layers.layer_norm import get_norm_cls
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.pynative.transformers.transformer_layer import BaseTransformerLayer
from mindformers.tools.logger import logger


@dataclass
class TransformerBlockSubmodules:
    """
    Class for specifying the submodules of a transformer block.

    This class defines the structure for configuring the layers and normalization
    within a transformer block, allowing for flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): Module specifications for
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
    """
    Transformer class.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.
        post_layer_norm (bool): Insert normalization layer at the end of transformer block. Default: True.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(S, B, H)` where S is sequence length,
          B is batch size, and H is hidden size.
        - **attention_mask** (Tensor) - Tensor of attention mask.
        - **rotary_pos_emb** (Tensor, optional) - Tensor of rotary position embedding. Default: None.
        - **prefix_keys_values** (optional) - List of prefix key-value tensors for each layer. Default: None.
        - **actual_seq_len** (optional) - Actual sequence length for variable-length sequences. Default: None.

    Outputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(S, B, H)`.
        - **extra_loss** (Tensor) - Extra loss value (e.g., from MoE layers). Default: 0.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            spec: Union[TransformerBlockSubmodules, ModuleSpec],
            post_layer_norm: bool = True,
    ):
        super().__init__()

        self.config = config
        self.submodules = _get_block_submodules(config, spec)
        self.post_layer_norm = post_layer_norm
        self.num_layers = config.num_layers
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        if config.sequence_parallel and cp > 1:
            logger.warning("The context parallel way conflicts with sequence parallel way. "
                           "The sequence parallel way has no effect and ignored.")
        self.seq_length_in_cfg = config.seq_length

        self._build_layers(config)

        self.init_extra_loss = Tensor([0], mstype.float32)

    def _build_layers(self, config: TransformerConfig):
        """build transformer layers."""
        # Transformer layers.
        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            layer = build_module(self.submodules.layer_specs[layer_id], config=config, layer_number=layer_id)
            self.layers.append(layer)

        if self.post_layer_norm:
            self.final_layernorm = build_module(self.submodules.layer_norm,
                                                dim=config.hidden_size,
                                                eps=config.layernorm_epsilon,
                                                params_dtype=config.params_dtype,
                                                compute_dtype=config.layernorm_compute_dtype)
        else:
            self.final_layernorm = None

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def construct(self,
                  hidden_states: Tensor,
                  attention_mask: Tensor,
                  rotary_pos_emb: Tensor = None,
                  prefix_keys_values=None,
                  actual_seq_len=None):
        """
        Construct function of transformer block.

        Args:
            hidden_states (Tensor): Input tensor of shape (S, B, H) where S is sequence length,
                B is batch size, and H is hidden size.
            attention_mask (Tensor): Attention mask tensor.
            rotary_pos_emb (Tensor, optional): Rotary position embedding tensor. Default: None.
            prefix_keys_values (optional): List of prefix key-value tensors for each layer.
                Each element should be a tuple or list of (key, value) tensors. Default: None.
            actual_seq_len (optional): Actual sequence length for variable-length sequences. Default: None.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - hidden_states (Tensor): Output tensor of shape (S, B, H).
                - extra_loss (Tensor): Extra loss value (e.g., from MoE layers).
        """
        extra_loss = self.init_extra_loss
        for index in range(self.num_layers):
            layer = self._get_layer(index)
            prefix_kv = prefix_keys_values[index] if prefix_keys_values is not None else None
            hidden_states, _, extra_loss = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                prefix_keys_values=prefix_kv,
                extra_loss=extra_loss,
                actual_seq_len=actual_seq_len
            )

        # final layernorm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, extra_loss
