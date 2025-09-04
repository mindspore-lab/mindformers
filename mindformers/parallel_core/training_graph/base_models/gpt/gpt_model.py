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

from typing import Literal, Optional, Union
import numpy as np

import mindspore as ms
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops import auto_generate as aclnn_ops
from mindspore.ops.operations import Morph
from mindspore import Tensor, dtype, nn
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.parallel_core.training_graph.loss_func import CrossEntropyLoss
from mindformers.parallel_core.training_graph.transformer.multi_token_prediction import MultiTokenPredictionBlock
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.parallel_core.training_graph.transformer.mask_generate import CausalMaskGenerate
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.base_models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding
)
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding
)
from mindformers.parallel_core.training_graph.base_models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding
)
from mindformers.parallel_core.training_graph.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSubmodules
)
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.tools.logger import logger
from mindformers.models.utils import get_current_rank_stage, get_model_parameters
from mindformers.version_control import get_lazy_inline as lazy_inline


def func_infer_dtype(*args):
    """infer_dtype for Morph."""
    return args[0]


def func_infer_shape(*args):
    """infer_shape for Morph."""
    input_shape = args[0]
    shape_value = np.prod(input_shape[:-1])
    output_shape = [int(shape_value), args[0][-1]]
    return output_shape


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
            config: TransformerConfig,
            transformer_layer_spec: Union[TransformerBlockSubmodules, ModuleSpec],
            vocab_size: int,
            max_sequence_length: int,
            pre_process: bool = True,
            post_process: bool = True,
            fp16_lm_cross_entropy: bool = False,
            parallel_output: bool = True,
            share_embeddings_and_output_weights: bool = False,
            position_embedding_type: Literal['learned_absolute', 'rope', 'yarn', 'none'] = 'learned_absolute',
            rotary_percent: float = 1.0,
            rotary_base: int = 10000,
            rope_scaling: bool = False,
            rope_scaling_factor: float = 8.0,
            scatter_embedding_sequence_parallel: bool = False,
            seq_len_interpolation_factor: Optional[float] = None,
            mtp_block_spec: ModuleSpec = None,
    ):
        super().__init__()
        if scatter_embedding_sequence_parallel:
            raise NotImplementedError('scatter_embedding_sequence_parallel is not supported for now.')

        self.config = config
        self.transformer_layer_spec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.is_zbv = ms.get_auto_parallel_context("pipeline_scheduler") == "zero_bubble_v"

        # init layout
        layout.init_layout(config)

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
        use_attn_mask_compression = config.use_attn_mask_compression or config.use_eod_attn_mask_compression

        # init parallel state groups
        self.tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
        self.dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        self.pp = config.pipeline_model_parallel_size if config.pipeline_model_parallel_size is not None else 1
        initialize_model_parallel(tensor_model_parallel_size=self.tp, data_parallel_size=self.dp,
                                  pipeline_model_parallel_size=self.pp)

        # Internally generates AttentionMask.
        self.casual_mask = CausalMaskGenerate(
            seq_length=config.seq_length,
            compute_type=config.compute_dtype,
            is_dynamic=config.is_dynamic,
            pad_token_id=config.pad_token_id,
            use_flash_attention=config.use_flash_attention,
            use_attn_mask_compression=use_attn_mask_compression,
            config=config
        )

        # Embeddings
        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel
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
        if self.use_rotary_position_embeddings:
            self.rotary_pos_emb.shard(config)
        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=False,
            post_process=False,
            # pre_process/post_process=True is not supported in TransformerBlock.
            # The corresponding Megatron v0.12.0 module's forward pass has this logic disabled by default,
            # so it won't cause significant impact.
        )

        # multi token prediction block
        self.init_mtp_loss = Tensor([0], ms.float32)
        self.init_numerator1 = Tensor([0], ms.float32)
        self.init_denominator1 = Tensor([1e-9], ms.float32)
        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(config=self.config, spec=mtp_block_spec)

        if self.post_process and fp16_lm_cross_entropy:
            raise ValueError("GPTModel does not need to support fp16_lm_cross_entropy.")
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
                                                     skip_weight_param_allocation=skip_weight_param_allocation)
            config.model_parallel = config.tensor_model_parallel_size
            self.loss = CrossEntropyLoss(config=config)

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
        self.assign = aclnn_ops.Assign()

        # init morphed layer
        self.morphed_reshape = Morph(self.forward_func, func_infer_shape, func_infer_dtype).add_prim_attr(
            "self_define_shard", True)

        # update topk-bias
        if config.moe_router_enable_expert_bias:
            if not config.num_moe_experts:
                config.moe_router_enable_expert_bias = False
                logger.warning("moe_router_enable_expert_bias is enabled, but num_moe_experts is 0 or not set. "
                               "Reset moe_router_enable_expert_bias to False.")
            else:
                self.moe_router_bias_update_rate = config.moe_router_bias_update_rate
                self.step_over_expert_num = \
                    Tensor([config.micro_batch_num / config.num_moe_experts], ms.float32)
                self.zeros_tensor = ms.Tensor(np.zeros([config.num_moe_experts]), ms.float32)

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
        if not self.config.use_eod_reset:
            position_ids = None
        elif position_ids is None:
            raise ValueError("When use eod_reset, position_ids should not be None.")
        if actual_seq_len is not None:
            actual_seq_len = self.reshape(actual_seq_len, (-1,))
        extra_block_kwargs = extra_block_kwargs or {}

        # Mindspore support TND layout by using actual_seq_len,
        # which indicates the partial seq_lens of eod sequences for compression mask.
        # Check mindformers.dataset.blended_datasets.gpt_dataset._get_eod_attention_mask() for implement details.
        extra_block_kwargs['actual_seq_len'] = actual_seq_len

        tokens, labels, attention_mask, loss_mask = self._preprocess_input_labels_and_masks(
            input_ids, labels, attention_mask, loss_mask
        )
        hidden_states, rotary_pos_emb, extra_loss = self.language_model(
            tokens,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            prefix_keys_values=prefix_keys_values,
            **extra_block_kwargs,
        )

        # multi token prediction
        mtp_loss = self.init_mtp_loss
        numerator1, denominator1 = self.init_numerator1, self.init_denominator1
        if self.mtp_process:
            mtp_loss, extra_loss = self.mtp(
                tokens,
                position_ids,
                hidden_states,
                attention_mask,
                labels=labels.reshape_as(tokens),
                rotary_pos_emb=rotary_pos_emb,
                loss_mask=loss_mask.reshape_as(tokens),
                extra_block_kwargs=extra_block_kwargs,
                word_embeddings_weight=self.embedding.word_embeddings.weight,
                position_embeddings_weight=getattr(self.embedding.position_embeddings, 'weight', None),
                tokentype_embeddings_weight=getattr(self.embedding.tokentype_embeddings, 'weight', None),
                output_weight=self.output_layer.weight,
                extra_loss=extra_loss,
            )
            if self.calculate_per_token_loss:
                numerator1, denominator1 = mtp_loss

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
            logits = self.morphed_reshape(logits)
        logits = self.cast(logits, dtype.float32)

        if not self.training:
            return logits.contiguous()

        # labels origin shape is [b s], Transpose is not required.
        loss = self.compute_language_model_loss(labels, logits, loss_mask)

        if self.calculate_per_token_loss:
            numerator0, denominator0 = loss
            return numerator0, denominator0, numerator1, denominator1, extra_loss * denominator0
        return loss, mtp_loss, extra_loss

    def forward_func(self, input_):
        """Morphed forward."""
        output = self.reshape(input_, (-1, input_.shape[-1]))
        return output

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
            actual_seq_len
        )

        return hidden_states, rotary_pos_emb, extra_loss

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
        if loss_mask is None:
            loss_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), dtype.float32)
        label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), dtype.float32)
        loss_mask = self.mul(loss_mask, label_mask)
        loss_mask = self.reshape(loss_mask, (-1,))
        labels = self.reshape(labels, (-1,))
        if self.config.use_eod_attn_mask_compression or self.config.use_attn_mask_compression:
            attention_mask = self.casual_mask()
        elif attention_mask is None:
            attention_mask = self.casual_mask(tokens)
        return tokens, labels, attention_mask, loss_mask

    def update_topk_bias(self, gradient_accumulation_steps: int = 1):
        """
        Will be called by mindformer.core.callback.TopkBiasBalanceCallback to
        update topk bias and reset expert_load of router in MoELayers.
        """
        config = self.config
        if not config.moe_router_enable_expert_bias:
            return []

        def _update_expert_load(router, gradient_accumulation_steps):
            expert_load_data = router.expert_load.value()
            if expert_load_data.sum() > 0:
                err = F.sub(self.step_over_expert_num * gradient_accumulation_steps, expert_load_data)
                expert_bias_new = F.add(
                    router.expert_bias.value(),
                    F.mul(F.sign(err), self.moe_router_bias_update_rate)
                )
                self.assign(router.expert_bias, expert_bias_new)
                self.assign(router.expert_load, self.zeros_tensor)
            return expert_load_data

        num_layers = config.num_layers

        if config.first_k_dense_replace:
            moe_layer_pattern = [0] * config.first_k_dense_replace + \
                                [1] * (num_layers - config.first_k_dense_replace)
        elif isinstance(config.moe_layer_freq, int):
            moe_layer_pattern = [1 if (i % config.moe_layer_freq == 0) else 0 for i in range(num_layers)]
        else:
            moe_layer_pattern = config.moe_layer_freq

        mtp_num_layers = config.mtp_num_layers
        if moe_layer_pattern[-1] == 0:
            mtp_num_layers = 0
        expert_loads = []
        for i in range(num_layers + mtp_num_layers):
            if moe_layer_pattern[min(i, num_layers - 1)] == 0:
                continue
            if i < num_layers:
                router = self.decoder.layers[i].mlp.router
                expert_load_data = _update_expert_load(router, gradient_accumulation_steps)
                expert_loads.append((f"decoder.layers.{i}.mlp.router", expert_load_data))
            else:
                router = self.mtp.layers[i - num_layers].transformer_layer.mlp.router
                expert_load_data = _update_expert_load(router, gradient_accumulation_steps)
                expert_loads.append((f"mtp.layers.{i - num_layers}.transformer_layer.mlp.router", expert_load_data))
        return expert_loads

    def shard(self, config: TransformerConfig):
        """parallel shard."""
        dp = config.data_parallel_size
        tp = config.tensor_model_parallel_size
        cp = 1 if config is None else config.context_parallel_size

        slice_in_strategy = ((dp, 1),)
        self.slice.shard(in_strategy=slice_in_strategy)
        not_equal_in_strategy = ((dp, 1), ())
        self.not_equal.shard(in_strategy=not_equal_in_strategy)
        mul_in_strategy = ((dp, 1), (dp, 1))
        self.mul.shard(in_strategy=mul_in_strategy)
        self.concat_prefix.shard(((dp, 1, cp, 1), (dp, 1, cp, 1)))
        self.transpose.shard(((cp, dp, tp),))
        pipeline_stage = config.pipeline_model_parallel_size
        if pipeline_stage > 1:
            self.embedding.pipeline_stage = 0
            self.embedding.pipeline_segment = 0
            if self.is_zbv:
                self.output_layer.pipeline_stage = 0
                self.output_layer.pipeline_segment = 1
                if self.mtp_process:
                    self.mtp.pipeline_stage = 0
                    self.mtp.pipeline_segment = 1
            else:
                self.output_layer.pipeline_stage = pipeline_stage - 1
                if self.mtp_process:
                    self.mtp.pipeline_stage = pipeline_stage - 1
        self.morphed_reshape.shard(
            in_strategy=(
                layout("dp", "cp", "tp"),
            ),
            out_strategy=(
                layout("dp_cp", "tp"),
            )
        )

    def sharding_propagation(self, config: TransformerConfig):
        pass

    def get_model_parameters(self):
        """Get current rank trainable parameters in gpt model ."""
        params = set()
        current_pipeline_stage = get_current_rank_stage()
        if ms.get_auto_parallel_context('pipeline_stages') > 1:
            if current_pipeline_stage == self.output_layer.pipeline_stage:
                params.update(get_model_parameters(self.output_layer))
            if current_pipeline_stage == self.mtp.pipeline_stage:
                params.update(get_model_parameters(self.mtp))
            params.update(self.decoder.get_model_parameters())
            params.update(get_model_parameters(self.embedding))
        else:
            params.update(get_model_parameters(self))
        return params
