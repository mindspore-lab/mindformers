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
"""Multi-Token Prediction (MTP) module for parallel token prediction during training.

This module enables the model to predict multiple future tokens in a single forward pass,
enhancing training efficiency and context awareness. It can be integrated with
speculative decoding for faster inference by generating draft tokens in parallel.

Note: Typically used only during training (disabled at inference in DeepSeek-V3).
"""

__all__ = ['MultiTokenPredictionBlock', 'MultiTokenPredictionBlock',
           'MultiTokenPredictionLayer', 'MultiTokenPredictionLayerSubmodules',
           'MultiTokenPredictionBlockSubmodules', 'get_mtp_layer_spec']

from dataclasses import dataclass
from typing import Union, List, Optional, Literal

from mindspore import nn, Tensor
from mindspore import dtype as mstype
from mindspore.ops.auto_generate import Cast, Concat, Reshape, Shape, StridedSlice, Zeros, Transpose, OnesLike

from mindformers.parallel_core.training_graph.loss_func import VocabParallelCrossEntropy
from mindformers.parallel_core.training_graph.transformer.norm import get_norm_cls
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.utils import LayerSetting
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, VocabParallelEmbedding
from mindformers.parallel_core.training_graph.base_models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding)


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
            layer_number: int = 1,
    ):
        super().__init__()
        self.config = config
        self.submodules = submodules
        self.dtype = config.compute_dtype
        self.use_seq_parallel = config.sequence_parallel
        self.layer_number = layer_number

        self.enorm = build_module(
            self.submodules.enorm,
            config=config,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        self.hnorm = build_module(
            self.submodules.hnorm,
            config=config,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        self.eh_proj = build_module(
            self.submodules.eh_proj,
            self.config.hidden_size * 2,
            self.config.hidden_size,
            config=config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            # The gather_output/is_expert parameter is unnecessary.
            # tp/ep partitioning and communication of module parameters is implemented by MindSpore's shard mechanism,
            # requiring no awareness from upper layers.
            # Other similar invocations should follow this same interpretation.
        )

        self.transformer_layer = build_module(
            self.submodules.transformer_layer,
            config=self.config,
        )

        self.final_layernorm = build_module(
            self.submodules.layer_norm,
            config=self.config,
            dim=self.config.hidden_size,
            eps=self.config.layernorm_epsilon
        )

        self.concat = Concat(axis=-1)
        self.concat_mp = Concat(axis=-1)
        self.cast = Cast()
        self.reshape = Reshape()

        self.shard(config)

    def shard(self, config: TransformerConfig):
        """Set parallel strategy."""
        dp = self.config.data_parallel_size
        tp = self.config.tensor_model_parallel_size
        cp = self.config.context_parallel_size
        self.concat.shard(((cp, dp, 1), (cp, dp, 1)))
        self.concat_mp.shard(((dp, tp, 1), (dp, tp, 1)))
        if self.use_seq_parallel and cp == 1:
            self.enorm.shard(config, in_strategy=(tp, dp, 1))
            self.hnorm.shard(config, in_strategy=(tp, dp, 1))
            self.concat.shard(((tp, dp, 1), (tp, dp, 1)))
            self.eh_proj.matmul.shard(((dp * tp, 1), (1, 1)))
            self.final_layernorm.shard(config, in_strategy=(tp, dp, 1))

    def construct(self,
                  decoder_input: Tensor,
                  hidden_states: Tensor,
                  attention_mask: Tensor,
                  extra_loss=0.,
                  rotary_pos_emb: Tensor = None,
                  actual_seq_len: Tensor = None):
        """
        Perform the forward pass through the MTP layer.

        Args:
            hidden_states (Tensor): hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            decoder_input (Tensor): Input tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
                At the (k - 1)-th MTP module, the i-th element of decoder input is
                the embedding of (i + K)-th tocken.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """

        # Note: Not Support FP8 Training.

        decoder_input = self.enorm(decoder_input)
        hidden_states = self.hnorm(hidden_states)

        # At the (k - 1)-th MTP module, concatenates the i-th tocken's hidden_states
        # and the (i + K)-th tocken's embedding, and combine them with linear projection.
        hidden_states = self.concat((decoder_input, hidden_states))
        hidden_states, _ = self.eh_proj(hidden_states)
        hidden_states, _, extra_loss = self.transformer_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            extra_loss=extra_loss,
            rotary_pos_emb=rotary_pos_emb,
            actual_seq_len=actual_seq_len
        )

        # Layer norm before shared head layer.
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, extra_loss


@dataclass
class MultiTokenPredictionBlockSubmodules:
    """
    Dataclass for specifying the submodules of a multi token prediction block.

    This class defines the structure for configuring the layers, allowing for
    flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the multi token prediction block. Each specification typically
            defines a complete multi token prediction layer (e.g., shared embedding,
            projection matrix, transformer block, shared output head).
    """

    layer_specs: List[ModuleSpec] = None


def _get_mtp_block_submodules(
        spec: Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]) -> MultiTokenPredictionBlockSubmodules:
    """
    Retrieve or construct MultiTokenPredictionBlockSubmodules based on the provided specification.

    Args:
        spec (Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]): Specification for the
            multi token prediction block submodules.
            Can be either a MultiTokenPredictionBlockSubmodules instance or a ModuleSpec.

    Returns:
        MultiTokenPredictionBlockSubmodules: The submodules for the multi token prediction block.
    """

    # Transformer block submodules.
    if isinstance(spec, MultiTokenPredictionBlockSubmodules):
        return spec
    if isinstance(spec, ModuleSpec):
        if issubclass(spec.module, MultiTokenPredictionBlock):
            return spec.submodules
        raise Exception(f"specialize for {spec.module.__name__}.")
    raise Exception(f"specialize for {type(spec).__name__}.")


class MultiTokenPredictionBlock(nn.Cell):
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

    def __init__(self, config: TransformerConfig, spec: Union[ModuleSpec]):
        super().__init__()
        self.config = config
        self.submodules = _get_mtp_block_submodules(spec)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self._build_layers()
        if not self.layers:
            raise ValueError("MultiTokenPredictionBlock must have at least one layer.")

        self.init_extra_loss = Tensor([0], mstype.float32)
        self.compute_language_model_loss = VocabParallelCrossEntropy(
            parallel_config=config, calculate_per_token_loss=config.calculate_per_token_loss)

        self.embedding = MtpSharedLanguageModelEmbedding(
            config=self.config,
            vocab_size=self.config.vocab_size,
            max_sequence_length=self.config.seq_length,
            position_embedding_type=self.config.position_embedding_type
        )
        self.embedding.pipeline_stage = config.pipeline_model_parallel_size - 1

        self.output_layer = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.vocab_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_weight_param_allocation=True,
        )
        self.output_layer.pipeline_stage = config.pipeline_model_parallel_size - 1

        self.cast = Cast()
        self.concat_2d = Concat(axis=-1)
        self.shape = Shape()
        self.slice = StridedSlice()
        self.zeros_op = Zeros()
        self.transpose = Transpose()
        self.reshape = Reshape()
        self.ones_like = OnesLike()

        self.shard()

    def _build_layers(self):
        """Building MTP layers."""
        self.layers = nn.CellList()
        # layer setting, take mtp layers into total layers.
        self.layer_setting = LayerSetting(
            self.config.num_layers + len(self.submodules.layer_specs),
            self.config.offset,
            self.config,
            self.config.virtual_pipeline_model_parallel_size
        )
        for i, layer_spec in enumerate(self.submodules.layer_specs):
            mtp_layer = build_module(layer_spec, config=self.config)
            self.layer_setting(mtp_layer, self.config.num_layers + i)
            self.layers.append(mtp_layer)

    def shard(self):
        dp = self.config.data_parallel_size
        cp = self.config.context_parallel_size

        self.transpose.shard(((dp, cp),))
        self.slice.shard(((dp, 1),))
        self.concat_2d.shard(((dp, 1), (dp, 1)))
        self.zeros_op.shard(((dp, 1),))
        self.ones_like.shard(((dp, 1),))

    def roll_tensor(self, tensor):
        """implement roll with slice and pad."""
        bs, seq_len = self.shape(tensor)
        pad_zeros = self.zeros_op((bs, 1))
        tensor = self.slice(tensor, (0, 1), (bs, seq_len), (1, 1))
        tensor = self.concat_2d((tensor, self.cast(pad_zeros, tensor.dtype)))

        return tensor

    def construct(
            self,
            input_ids: Tensor,  # [b,s]
            position_ids: Tensor,  # [b,s]
            hidden_states: Tensor,  # [s,b,h]
            attention_mask: Tensor,  # [b,1,s,s]
            labels: Tensor = None,  # [b,s]
            rotary_pos_emb: Tensor = None,
            extra_block_kwargs: dict = None,
            loss_mask: Optional[Tensor] = None,  # [b,s]
            word_embeddings_weight: Tensor = None,
            position_embeddings_weight: Tensor = None,
            tokentype_embeddings_weight: Optional[Tensor] = None,
            output_weight: Tensor = None,
            extra_loss=None,
    ):
        """
        Perform the forward pass through all of the MTP modules.

        Args:
            hidden_states (Tensor): Hidden states for input token with the shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.

        Returns:
            (Tensor): The mtp loss tensor of shape [b, s].
        """
        if labels is None:
            raise ValueError("labels should not be None for calculating multi token prediction loss.")
        if loss_mask is None:
            # if loss_mask is not provided, use all ones as loss_mask
            loss_mask = self.ones_like(labels)
        if extra_loss is None:
            extra_loss = self.init_extra_loss

        mtp_loss = 0
        for layer in self.layers:
            # Calc logits for the current Multi-Token Prediction (MTP) layers.
            input_ids = self.roll_tensor(input_ids)
            # embedding
            decoder_input = self.embedding(
                input_ids=input_ids,
                position_ids=position_ids,
                word_embeddings_weight=word_embeddings_weight,
                position_embeddings_weight=position_embeddings_weight,
                tokentype_embeddings_weight=tokentype_embeddings_weight,
            )
            # norm, linear projection and transformer
            hidden_states, extra_loss = layer(
                decoder_input=decoder_input,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                extra_loss=extra_loss,
                rotary_pos_emb=rotary_pos_emb,
                **(extra_block_kwargs or {}),
            )
            # output
            mtp_logits, _ = self.output_layer(
                hidden_states, weight=output_weight
            )
            seq_len, bsz, _ = self.shape(mtp_logits)
            mtp_logits = self.reshape(mtp_logits, (seq_len * bsz, -1))

            # Calc loss for the current Multi-Token Prediction (MTP) layers.
            labels = self.roll_tensor(labels)
            loss_mask = self.roll_tensor(loss_mask)

            # If the compute_language_model_loss is actually unwrapped VocabParallelCrossEntropy, the inputs should
            # be reshaped manually.
            labels_t = self.reshape(self.transpose(labels, (1, 0)), (-1,))
            loss_mask_t = self.reshape(self.transpose(loss_mask, (1, 0)), (-1,))

            # config.calculate_per_token_loss is supported in training_graph.loss_func.VocabParallelCrossEntropy
            mtp_layer_loss = self.compute_language_model_loss(mtp_logits, labels_t, loss_mask_t)
            mtp_layer_loss_scale = self.mtp_loss_scaling_factor / self.config.mtp_num_layers
            mtp_layer_loss = mtp_layer_loss_scale * mtp_layer_loss
            # MTPLossAutoScaler is not supported for now, forward is not effective, backward grad scale=1.0 by default.
            mtp_loss = mtp_loss + mtp_layer_loss

        return mtp_loss, extra_loss


class MtpSharedVocabParallelEmbedding(VocabParallelEmbedding):
    """Embedding layer used in Multi-Token Prediction module, same to standard embedding."""

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 config: TransformerConfig):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            config=config,
            init_method=config.init_method,
        )
        # use shared embedding weights instead
        del self.weight

    def construct(self, weight, input_ids):
        """Forward of vocab embedding."""
        bs, seq_len = input_ids.shape
        # in IndexSelect, input_ids should be 1-dimension
        input_ids_ = self.reshape(input_ids, (bs * seq_len,))
        output_ = self.gather(weight, 0, input_ids_)
        output = self.reshape(output_, (bs, seq_len, -1))

        return output


class MtpSharedLanguageModelEmbedding(LanguageModelEmbedding):
    """Embedding layer used in Multi-Token Prediction module, same to standard LanguageModelEmbedding."""

    def __init__(
            self,
            config: TransformerConfig,
            vocab_size: int,
            max_sequence_length: int,
            position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
            num_tokentypes: int = 0
    ):
        super().__init__(
            config,
            vocab_size,
            max_sequence_length,
            position_embedding_type,
            num_tokentypes
        )
        # Word embedding
        self.word_embeddings = MtpSharedVocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=config.hidden_size,
            config=config
        )
        # Position embedding
        self.add_position_embedding = position_embedding_type == 'learned_absolute'
        if self.add_position_embedding:
            self.position_embeddings = MtpSharedVocabParallelEmbedding(
                num_embeddings=max_sequence_length,
                embedding_dim=config.hidden_size,
                config=config,
            )
        # tokentypes embedding
        if num_tokentypes > 0:
            self.tokentype_embeddings = MtpSharedVocabParallelEmbedding(
                num_embeddings=num_tokentypes,
                embedding_dim=config.hidden_size,
                config=config,
            )
        else:
            self.tokentype_embeddings = None

    def construct(
            self,
            input_ids,
            position_ids,
            word_embeddings_weight,
            position_embeddings_weight,
            tokentype_embeddings_weight,
            tokentype_ids=None
    ):
        """embedding construct"""
        words_embeddings = self.word_embeddings(word_embeddings_weight, input_ids)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_embeddings_weight, position_ids)
            embeddings = self.add_pe(words_embeddings, position_embeddings)
        else:
            embeddings = words_embeddings

        if tokentype_ids is not None:
            if self.tokentype_embeddings is None:
                raise RuntimeError("Embedding layer got 'tokentype_ids' input, "
                                   "but 'tokentype_embeddings' layer is not initialized")
            tokentype_embedding = self.tokentype_embeddings(tokentype_embeddings_weight, tokentype_ids)
            embeddings = self.add_te(embeddings, tokentype_embedding)
        else:
            if self.tokentype_embeddings is not None:
                raise RuntimeError("The 'tokentype_ids' input for Embedding layer is None, "
                                   "but 'tokentype_embeddings' layer is initialized")

        # Data format change to avoid explicit transposes : [b s h] --> [s b h].
        embeddings = self.transpose(embeddings, (1, 0, 2))

        # Dropout
        if self.embedding_dropout_prob > 0:
            embeddings = self.embedding_dropout(embeddings)

        embeddings = self.cast(embeddings, self.compute_dtype)
        return embeddings
