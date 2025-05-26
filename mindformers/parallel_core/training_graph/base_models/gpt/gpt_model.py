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
"""mindformers GPT model"""
__all__ = ['GPTModel']

from typing import Literal, Optional
from mindspore.ops import operations as P
from mindspore.ops import auto_generate as aclnn_ops
from mindspore import Tensor, dtype, nn
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindformers.parallel_core.training_graph.loss_func import CrossEntropyLoss
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.training_graph.transformer.mask_generate import CausalMaskGenerate
from mindformers.parallel_core.transformer_config import TransformerConfig, MLATransformerConfig
from mindformers.parallel_core.training_graph.base_models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding)
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding)
from mindformers.parallel_core.training_graph.base_models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding)
from mindformers.parallel_core.training_graph.transformer.transformer_block import TransformerBlock
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear
from mindformers.version_control import get_lazy_inline as lazy_inline


class GPTModel(nn.Cell):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig):
            Transformer config.
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers.
        vocab_size (int):
            Vocabulary size.
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding.
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional):
            Defaults to False.
        parallel_output (bool, optional):
            Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        rope_scaling (bool, optional): Toggle RoPE scaling.
        rope_scaling_factor (float): RoPE scaling factor. Default 8.
        scatter_embedding_sequence_parallel (bool, optional):
            Whether embeddings should be scattered across sequence parallel
            region or not. Defaults to True.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
    """

    @lazy_inline
    def __init__(
            self,
            config: MLATransformerConfig,
            transformer_layer_spec: ModuleSpec,
            vocab_size: int,
            max_sequence_length: int,
            pre_process: bool = True,
            post_process: bool = True,
            fp16_lm_cross_entropy: bool = False,
            parallel_output: bool = True,
            share_embeddings_and_output_weights: bool = False,
            position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
            rotary_percent: float = 1.0,
            rotary_base: int = 10000,
            rope_scaling: bool = False,
            rope_scaling_factor: float = 8.0,
            scatter_embedding_sequence_parallel: bool = False,
            seq_len_interpolation_factor: Optional[float] = None,
            mtp_layer_spec: ModuleSpec = None,
    ):
        super().__init__()
        if scatter_embedding_sequence_parallel:
            raise NotImplementedError('scatter_embedding_sequence_parallel is not supported for now.')

        self.config = config
        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        if hasattr(self.config, 'position_embedding_type'):
            # By default, use the position_embedding_type configuration in TransformerConfig.
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type

        self.rotary_percent = rotary_percent
        if hasattr(self.config, 'rotary_base'):
            # By default, use the rotary_base configuration in TransformerConfig.
            self.rotary_base = self.config.rotary_base
        else:
            self.rotary_base = rotary_base

        self.rotary_scaling = rope_scaling
        self.mtp_block_spec = mtp_layer_spec
        self.mtp_process = mtp_layer_spec is not None

        self.rope_scaling_factor = rope_scaling_factor
        self.rotary_seq_len_interpolation_factor = seq_len_interpolation_factor \
            if seq_len_interpolation_factor is not None else config.rotary_seq_len_interpolation_factor

        # get value from config
        self.use_eod_attn_mask_compression = config.use_eod_attn_mask_compression
        self.init_method = config.init_method
        self.compute_dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.pad_token_id = config.pad_token_id
        self.ignore_token_id = config.ignore_token_id

        # Internally generates AttentionMask.
        self.casual_mask = CausalMaskGenerate(
            seq_length=config.seq_length,
            compute_type=config.compute_dtype,
            is_dynamic=config.is_dynamic,
            pad_token_id=config.pad_token_id,
            use_flash_attention=config.use_flash_attention,
            use_attn_mask_compression=config.use_eod_attn_mask_compression,
            config=config
        )

        # Embeddings
        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel)

        # rope
        # The MTP implementation pre-computes RotaryEmbedding
        # (unlike Megatron's real-time generation) to minimize dynamic memory usage.
        self.use_rotary_position_embeddings = self.position_embedding_type in ['rope', 'yarn']
        if self.position_embedding_type == 'rope':
            if config.multi_latent_attention:
                self.rotary_pos_emb = RotaryEmbedding(
                    kv_channels=config.qk_pos_emb_head_dim,
                    rotary_percent=config.rotary_percent,
                    rotary_base=config.rotary_base,
                )
            else:
                self.seq_length = config.seq_length
                self.rotary_pos_emb = RotaryEmbedding(
                    kv_channels=self.config.kv_channels,
                    rotary_percent=rotary_percent,
                    rotary_interleaved=self.config.rotary_interleaved,
                    seq_len_interpolation_factor=seq_len_interpolation_factor,
                    rotary_base=rotary_base,
                    rope_scaling=rope_scaling,
                    rope_scaling_factor=rope_scaling_factor,
                )
        elif self.position_embedding_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                kv_channels=config.qk_pos_emb_head_dim,
                rotary_base=config.rotary_base,
                scaling_factor=config.rotary_scaling_factor,
                original_max_position_embeddings=config.max_position_embeddings,
                beta_fast=config.beta_fast,
                beta_slow=config.beta_slow,
                mscale=config.mscale,
                mscale_all_dim=config.mscale_all_dim,
            )
        elif self.position_embedding_type == 'mrope':
            raise NotImplementedError("position_embedding_type=mrope not support.")

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=False,
            post_process=False,
            # pre_process/post_process=True is not supported in TransformerBlock.
            # The corresponding Megatron module's forward pass has this logic disabled by default,
            # so it won't cause significant impact.
        )

        # multi token prediction block
        if self.mtp_process:
            self.mtp = build_module(mtp_layer_spec.module, config=config, submodules=mtp_layer_spec.submodules)
            # remove parameters name prefix 'mtp_block'
            self.mtp.update_parameters_name(prefix='')

        # Output
        if self.post_process or self.mtp_process:
            skip_weight_param_allocation = self.pre_process and self.share_embeddings_and_output_weights
            self.output_layer = ColumnParallelLinear(input_size=self.hidden_size,
                                                     output_size=self.vocab_size,
                                                     config=config,
                                                     init_method=self.init_method,
                                                     bias=False,
                                                     skip_bias_add=False,
                                                     gather_output=not self.parallel_output,
                                                     skip_weight_param_allocation=skip_weight_param_allocation,
                                                     )
            config.model_parallel = config.tensor_model_parallel_size
            self.loss = CrossEntropyLoss(parallel_config=config)

        # operations
        self.cast = aclnn_ops.Cast()
        self.concat_prefix = aclnn_ops.Concat(-1)
        self.zeros = P.Zeros()
        self.shape = aclnn_ops.Shape()
        self.slice = aclnn_ops.StridedSlice()
        self.not_equal = P.NotEqual()
        self.reshape = aclnn_ops.Reshape()
        self.mul = aclnn_ops.Mul()
        self.add = aclnn_ops.AddExt()
        self.transpose = aclnn_ops.Transpose()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        else:
            self.shard(config)

    def construct(
            self,
            input_ids: Tensor,
            position_ids: Tensor = None,
            attention_mask: Tensor = None,
            decoder_input: Tensor = None,
            labels: Tensor = None,
            extra_block_kwargs=None,
            prefix_keys_values=None,
            loss_mask=None,
            actual_seq_len=None
    ):
        """GPTModel construct"""
        tokens, labels, attention_mask, loss_mask = self._preprocess_input_labels_and_masks(
            input_ids, labels, attention_mask, loss_mask
        )
        hidden_states, extra_loss = self.language_model(
            tokens,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            prefix_keys_values=prefix_keys_values,
            actual_seq_len=actual_seq_len,
            **(extra_block_kwargs or {}),
        )

        # multi token prediction
        if self.mtp_process:
            rotary_pos_emb = None
            if self.use_rotary_position_embeddings:
                rotary_pos_emb = self.rotary_pos_emb(self.seq_length, position_ids=position_ids)
            mtp_loss, extra_loss = self.mtp(
                mtp_tokens=tokens,
                hidden_states=hidden_states,
                shared_emb_weight=self.embedding.word_embeddings.weight,
                shared_head_weight=self.output_layer.weight,
                mtp_labels=labels.reshape_as(tokens),
                mtp_loss_mask=loss_mask.reshape_as(tokens),
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                extra_loss=extra_loss,
                prefix_keys_values=prefix_keys_values,
            )
            extra_loss = self.add(extra_loss, mtp_loss)

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if not self.post_process:
            return hidden_states

        # labels origin shape is [b s h], Transpose is not required.
        loss = self.compute_language_model_loss(hidden_states, labels, output_weight,
                                                self.fp16_lm_cross_entropy, loss_mask)
        # moe/mtp extra loss, default 0.0
        loss = self.add(loss, extra_loss)
        return loss

    def language_model(
            self,
            input_ids,
            position_ids,
            attn_mask,
            decoder_input,
            tokentype_ids=None,
            prefix_keys_values=None,
            actual_seq_len=None
    ):
        """decoder output"""
        bs, seq_len = self.shape(input_ids)
        if actual_seq_len is not None:
            actual_seq_len = self.reshape(actual_seq_len, (-1,))
        # Encoder embedding
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids, position_ids, tokentype_ids=tokentype_ids)
        else:
            decoder_input = None

        # rope
        rotary_pos_emb = None
        if self.use_rotary_position_embeddings:
            rotary_pos_emb = self.rotary_pos_emb(seq_len)

        if prefix_keys_values is not None:
            if attn_mask is None:
                raise ValueError("attn_mask should not be None when prefix_keys_values is not None!")
            if self.config.use_attn_mask_compression or attn_mask.ndim != 4:
                raise ValueError("use_attn_mask_compression should be False when prefix_keys_values is not None! "
                                 "And attn_mask.ndim should be 4, but got {}".format(attn_mask.ndim))

            # prefix_key_values shape num_layers*(2, B, prefix_len, kv_num*kv_channel)
            bs, seq_len = self.shape(input_ids)
            prefix_length = self.shape(prefix_keys_values[0])[2]
            prefix_mask = self.zeros((bs, 1, seq_len, prefix_length), attn_mask.dtype)
            # (B, 1, S, S) -> (B, 1, S, S+prefix_len)
            attn_mask = self.concat_prefix((prefix_mask, attn_mask))

        # Run decoder.
        hidden_states, extra_loss = self.decoder(
            decoder_input,
            attn_mask,
            rotary_pos_emb,
            prefix_keys_values,
        )

        return hidden_states, extra_loss

    def shared_embedding_or_output_weight(self):
        """Gets the emedding weight or output logit weights when share embedding and output weights set to True.

        Returns:
            Tensor: During pre processing it returns the input embeddings weight while during post processing
            it returns the final output layers weight
        """
        if self.pre_process:
            return self.embedding.word_embeddings.weight
        if self.post_process:
            return self.output_layer.weight
        return None

    def compute_language_model_loss(self,
                                    lm_output: Tensor,
                                    labels: Tensor,
                                    logit_weights: Tensor,
                                    fp16_lm_cross_entropy: bool,
                                    loss_mask: Tensor
                                    ):
        """Post-processing of language model output.

        Args:
            lm_output (Tensor): Language model output.
            labels (Tensor): Labels.
            logit_weights (Tensor): Logit weights.
            fp16_lm_cross_entropy (bool): Whether to use fp16 for loss computation.
            loss_mask (Tensor): Loss mask.

        Returns:
            output (Tensor): Output loss.
        """
        if fp16_lm_cross_entropy:
            raise ValueError("GPTModel does not need to support fp16_lm_cross_entropy.")
        logits, _ = self.output_layer(lm_output, logit_weights)

        if not self.training:
            return logits.contiguous()

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        output = self.cast(logits, dtype.float32)
        output = self.transpose(output, (0, 1))
        return self.loss(output, labels, loss_mask)

    def _preprocess_input_labels_and_masks(self,
                                           input_ids: Tensor,
                                           labels: Tensor = None,
                                           attention_mask: Tensor = None,
                                           loss_mask: Tensor = None):
        """Preprocess input_ids and generate labels and masks if they are None.

        Args:
            labels (Tensor): Labels.
            attention_mask (Tensor): Attention mask.

        Returns:
            tokens (Tensor): Processed tokens if in training.
            labels (Tensor): Labels if input is none.
            attention_mask (Tensor): Attention mask if input is none.
            loss_mask (Tensor): Loss mask.
        """
        tokens = input_ids
        label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), dtype.float32)
        loss_mask = self.mul(loss_mask, label_mask)
        loss_mask = self.reshape(loss_mask, (-1,))
        labels = self.reshape(labels, (-1,))
        if self.config.use_eod_attn_mask_compression:
            attention_mask = self.casual_mask()
        elif attention_mask is None:
            attention_mask = self.casual_mask(tokens)
        return tokens, labels, attention_mask, loss_mask

    def shard(self, config: TransformerConfig):
        """parallel shard."""
        dp = config.data_parallel_size
        cp = 1 if config is None else config.context_parallel_size
        slice_in_strategy = ((dp, 1),)
        self.slice.shard(in_strategy=slice_in_strategy)
        not_equal_in_strategy = ((dp, 1), ())
        self.not_equal.shard(in_strategy=not_equal_in_strategy)
        mul_in_strategy = ((dp, 1), (dp, 1))
        self.mul.shard(in_strategy=mul_in_strategy)
        self.concat_prefix.shard(((dp, 1, cp, 1), (dp, 1, cp, 1)))
        self.transpose.shard(((cp, dp),))
        pipeline_stage = config.pipeline_model_parallel_size
        if pipeline_stage > 1:
            self.embedding.pipeline_stage = 0
            self.output_layer.pipeline_stage = pipeline_stage - 1

    def sharding_propagation(self, config: TransformerConfig):
        pass
