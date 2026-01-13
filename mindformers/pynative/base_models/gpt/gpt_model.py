# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Modification points:
# 1. Replace all interfaces with MindSpore TransFormers'.
# 2. Add some input parameters for MindSpore TransFormers.
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

from typing import Literal, Optional, Union

from mindspore import Tensor, dtype, nn, mint, ops

from mindformers.pynative.loss.loss import CrossEntropyLoss
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.pynative.layers.mask_generate import CausalMaskGenerate
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.base_models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from mindformers.pynative.base_models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from mindformers.pynative.base_models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from mindformers.pynative.transformers.transformer_block import TransformerBlock, TransformerBlockSubmodules
from mindformers.pynative.layers.linear import Linear


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
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        rope_scaling (bool, optional): Toggle RoPE scaling. Defaults to False.
        rope_scaling_factor (float): RoPE scaling factor. Defaults to 8.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
        mtp_block_spec (ModuleSpec): A mtp block spec. Defaults to None.
    """

    def __init__(
            self,
            config: TransformerConfig,
            transformer_layer_spec: Union[TransformerBlockSubmodules, ModuleSpec],
            vocab_size: int,
            max_sequence_length: int,
            pre_process: bool = True,
            post_process: bool = True,
            share_embeddings_and_output_weights: bool = False,
            position_embedding_type: Literal['learned_absolute', 'rope', 'yarn', 'none'] = 'learned_absolute',
            rotary_percent: float = 1.0,
            rotary_base: int = 10000,
            rope_scaling: bool = False,
            rope_scaling_factor: float = 8.0,
            seq_len_interpolation_factor: Optional[float] = None,
            mtp_block_spec: ModuleSpec = None,
    ):
        super().__init__()

        self.config = config
        self.transformer_layer_spec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.use_attn_mask_compression = config.use_attn_mask_compression or config.use_eod_attn_mask_compression

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
        self.rope_scaling_factor = rope_scaling_factor
        self.rotary_seq_len_interpolation_factor = seq_len_interpolation_factor \
            if seq_len_interpolation_factor is not None else config.rotary_seq_len_interpolation_factor
        self.seq_length = config.seq_length
        self.mtp_process = mtp_block_spec is not None

        # get value from config
        self.use_eod_attn_mask_compression = config.use_eod_attn_mask_compression
        self.init_method = config.init_method
        self.compute_dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.pad_token_id = config.pad_token_id
        self.ignore_token_id = config.ignore_token_id
        self.calculate_per_token_loss = config.calculate_per_token_loss

        # Internally generates AttentionMask.
        self.casual_mask = CausalMaskGenerate(
            seq_length=config.seq_length,
            compute_type=config.compute_dtype,
            is_dynamic=config.is_dynamic,
            pad_token_id=config.pad_token_id,
            use_attn_mask_compression=self.use_attn_mask_compression,
        )

        # Embeddings
        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
            )

        # rope
        # The MTP implementation pre-computes RotaryEmbedding
        # (unlike Megatron v0.12.0's real-time generation) to minimize dynamic memory usage.
        self.use_rotary_position_embeddings = self.position_embedding_type in ['rope', 'yarn']
        if self.position_embedding_type == 'rope':
            if config.multi_latent_attention:
                self.rotary_pos_emb = RotaryEmbedding(
                    kv_channels=config.qk_pos_emb_head_dim,
                    rotary_percent=config.rotary_percent,
                    rotary_base=config.rotary_base,
                    use_eod_reset=config.use_eod_reset
                )
            else:
                self.rotary_pos_emb = RotaryEmbedding(
                    kv_channels=self.config.kv_channels,
                    rotary_percent=rotary_percent,
                    rotary_interleaved=self.config.rotary_interleaved,
                    seq_len_interpolation_factor=seq_len_interpolation_factor,
                    rotary_base=rotary_base,
                    rope_scaling=rope_scaling,
                    rope_scaling_factor=rope_scaling_factor,
                    use_eod_reset=config.use_eod_reset
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
                use_eod_reset=config.use_eod_reset
            )
        elif self.position_embedding_type == 'mrope':
            raise NotImplementedError("position_embedding_type = mrope is not supported now.")

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            # The corresponding Megatron v0.12.0 module's forward pass has this logic disabled by default,
            # so it won't cause significant impact.
        )

        # Output
        if self.post_process or self.mtp_process:
            skip_weight_param_allocation = self.pre_process and self.share_embeddings_and_output_weights
            self.output_layer = Linear(input_size=self.hidden_size,
                                       output_size=self.vocab_size,
                                       init_method=self.init_method,
                                       bias=False,
                                       skip_bias_add=False,
                                       skip_weight_param_allocation=skip_weight_param_allocation,
                                       compute_dtype=self.config.compute_dtype,
                                       params_dtype=self.config.params_dtype)
            config.model_parallel = config.tensor_model_parallel_size
            self.loss = CrossEntropyLoss(config=config)

        # operations
        self.cast = ops.cast
        self.concat_prefix = mint.concat
        self.zeros = mint.zeros
        self.not_equal = mint.not_equal
        self.reshape = mint.reshape
        self.mul = mint.mul
        self.add = mint.add
        self.sub = mint.sub
        self.sign = mint.sign
        self.transpose = mint.permute
        self.assign = mint.clone

    def construct(
            self,
            input_ids: Tensor,
            position_ids: Tensor = None,
            attention_mask: Tensor = None,
            decoder_input: Tensor = None,
            labels: Tensor = None,
            loss_mask=None,
            actual_seq_len=None,
    ):
        """GPTModel construct.

        Args:
            input_ids (Tensor): The input tensor of token IDs.
            position_ids (Tensor, optional): Position ID tensor, used to specify the position
            of each token. Default is None.
            attention_mask (Tensor, optional): Attention mask tensor, used to mask padding
            tokens. Default is None.
            decoder_input (Tensor, optional): Decoder input tensor. Default is None.
            labels (Tensor, optional): The label tensor, used for calculating the loss. 
            Default is None.
            loss_mask (Tensor, optional): Loss mask tensor, used to specify which positions
            are included in the loss calculation. Default is None.
            actual_seq_len (Tensor, optional): Actual sequence length tensor. Default is None.
        """
        if not self.config.use_eod_reset:
            position_ids = None
        elif position_ids is None:
            raise ValueError("When use eod_reset, position_ids should not be None.")
        if actual_seq_len is not None:
            actual_seq_len = self.reshape(actual_seq_len, (-1,))

        # Mindspore support TND layout by using actual_seq_len,
        # which indicates the partial seq_lens of eod sequences for compression mask.
        # Check mindformers.dataset.blended_datasets.gpt_dataset._get_eod_attention_mask() for implement details.

        labels, attention_mask, loss_mask = self._preprocess_input_labels_and_masks(
            input_ids, labels, attention_mask, loss_mask)

        hidden_states, _, extra_loss = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            actual_seq_len=actual_seq_len
        )

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if not self.post_process:
            return hidden_states

        # logits origin shape is [s b h], transform it to [b*s h].
        logits, _ = self.output_layer(hidden_states, output_weight)
        if logits.ndim > 2:
            logits = self.transpose(logits, (1, 0, 2))
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, dtype.float32)

        if not self.training:
            return logits.contiguous()

        # labels origin shape is [b s], Transpose is not required.
        loss = self.compute_language_model_loss(labels, logits, loss_mask)

        if self.calculate_per_token_loss:
            numerator0, denominator0 = loss
            return numerator0, denominator0, extra_loss * denominator0
        return loss, extra_loss, logits, hidden_states

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
        """decoder output.

        Args:
            input_ids (Tensor): The input tensor of token IDs.
            position_ids (Tensor, optional): Position ID tensor, used to specify the position
            of each token. Default is None.
            attn_mask (Tensor, optional): Attention mask tensor, used to mask padding
            tokens. Default is None.
            decoder_input (Tensor, optional): Decoder input tensor. Default is None.
            tokentype_ids (Tensor, optional): Token's type ID. Default is None.
            prefix_keys_values (Tensor, optional): Prefix key-value pairs, used for
            scenarios such as prefix tuning. The default value is None.
            actual_seq_len (Tensor, optional): Actual sequence length tensor. Default is None.
        """
        bs, seq_len = input_ids.shape
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
            rotary_pos_emb = self.rotary_pos_emb(seq_len, position_ids=position_ids)

        if prefix_keys_values is not None:
            if attn_mask is None:
                raise ValueError("attn_mask should not be None when prefix_keys_values is not None!")
            if self.config.use_attn_mask_compression or attn_mask.ndim != 4:
                raise ValueError("use_attn_mask_compression should be False when prefix_keys_values is not None! "
                                 f"And attn_mask.ndim should be 4, but got {attn_mask.ndim}")

            # prefix_key_values shape num_layers*(2, B, prefix_len, kv_num*kv_channel)
            bs, seq_len = input_ids.shape
            prefix_length = prefix_keys_values[0].shape[2]
            prefix_mask = self.zeros((bs, 1, seq_len, prefix_length), attn_mask.dtype)
            # (B, 1, S, S) -> (B, 1, S, S+prefix_len)
            attn_mask = self.concat_prefix((prefix_mask, attn_mask))

        # Run decoder.
        hidden_states, extra_loss = self.decoder(
            decoder_input,
            attn_mask,
            rotary_pos_emb,
            prefix_keys_values,
            actual_seq_len
        )

        return hidden_states, rotary_pos_emb, extra_loss

    def shared_embedding_or_output_weight(self):
        """Gets the embedding weight or output logit weights when share embedding and output weights set to True.

        Returns:
            Tensor: During pre-processing it returns the input embeddings weight while during post-processing
            it returns the final output layers weight
        """
        if self.pre_process:
            return self.embedding.word_embeddings.weight
        if self.post_process:
            return self.output_layer.weight
        return None

    def compute_language_model_loss(self,
                                    labels: Tensor,
                                    logits: Tensor,
                                    loss_mask: Tensor
                                    ):
        """Post-processing of language model output.

        Args:
            labels (Tensor): Labels.
            logits (Tensor): Logit.
            loss_mask (Tensor): Loss mask.

        Returns:
            output (Tensor): Output loss.
        """
        return self.loss(logits, labels, loss_mask)

    def _preprocess_input_labels_and_masks(self,
                                           input_ids: Tensor,
                                           labels: Tensor = None,
                                           attention_mask: Tensor = None,
                                           loss_mask: Tensor = None):
        """Preprocess input_ids and generate labels and masks if they are None.
        """
        if loss_mask is None:
            loss_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), dtype.float32)
        label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), dtype.float32)
        loss_mask = self.mul(loss_mask, label_mask)
        if self.use_attn_mask_compression:
            attention_mask = self.casual_mask()
        elif attention_mask is None:
            attention_mask = self.casual_mask(input_ids)
        return labels, attention_mask, loss_mask
