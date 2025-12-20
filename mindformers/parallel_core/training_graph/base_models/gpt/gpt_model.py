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
from mindspore.communication import get_group_size, get_rank
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops import auto_generate as aclnn_ops
from mindspore.ops.operations import Morph
from mindspore import Tensor, dtype, nn
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore import ops

from mindformers.parallel_core.training_graph.loss_func import CrossEntropyLoss
from mindformers.parallel_core.training_graph.transformer.multi_token_prediction import MultiTokenPredictionBlock, \
    func_infer_dtype, func_infer_shape, func_infer_shape_labels_and_masks
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.training_graph.communication import (
    compute_repeat_num_and_model_parallel_size,
    get_cp_group_name,
    get_dp_group_name,
    get_op_group_name
)
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
from mindformers.parallel_core.utils.init_method import init_method_normal
from mindformers.tools.logger import logger
from mindformers.models.utils import get_current_rank_stage, get_model_parameters
from mindformers.version_control import get_lazy_inline as lazy_inline
from mindformers.core.optim.muon_utils import make_muon_fns

class PreprocessLabelsAndMasks(nn.Cell):
    """Preprocess input_ids and generate labels and masks.
    """
    def __init__(self, config):
        super().__init__()
        self.use_attn_mask_compression = config.use_attn_mask_compression or config.use_eod_attn_mask_compression

        # Operations
        self.cast = aclnn_ops.Cast()
        self.not_equal = P.NotEqual()
        self.mul = aclnn_ops.Mul()
        self.shape = aclnn_ops.Shape()
        self.reshape = aclnn_ops.Reshape()
        self.pad_token_id = config.pad_token_id
        self.ignore_token_id = config.ignore_token_id

         # Internally generates AttentionMask.
        self.casual_mask = CausalMaskGenerate(
            seq_length=config.seq_length,
            compute_type=config.compute_dtype,
            is_dynamic=config.is_dynamic,
            pad_token_id=config.pad_token_id,
            use_flash_attention=config.use_flash_attention,
            use_attn_mask_compression=self.use_attn_mask_compression,
            config=config
        )
        self.morphed_reshape_labels_and_masks = Morph(
            self.forward_func_labels_and_masks,
            func_infer_shape_labels_and_masks,
            func_infer_dtype
        ).add_prim_attr("self_define_shard", True)

        self.shard()

    def forward_func_labels_and_masks(self, input_):
        """Morphed forward."""
        output = self.reshape(input_, (-1,))
        return output

    def shard(self):
        self.morphed_reshape_labels_and_masks.shard(
            in_strategy=(layout("dp", "cp"),),
            out_strategy=(layout("dp_cp"),))


    def construct(self, input_ids: Tensor,
                  labels: Tensor = None,
                  attention_mask: Tensor = None,
                  loss_mask: Tensor = None):
        """
        Preprocess input_ids and generate labels and masks if they are None.

        Args:
            labels (Tensor): Labels.
            attention_mask (Tensor): Attention mask.
            loss_mask (Tensor): Label mask
        Returns:
            labels (Tensor): Labels if input is none.
            attention_mask (Tensor): Attention mask if input is none.
            loss_mask (Tensor): Loss mask.
        """
        if loss_mask is None:
            loss_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), dtype.float32)

        if labels is not None:
            label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), dtype.float32)
            loss_mask = self.mul(loss_mask, label_mask)
            local_loss_mask = self.morphed_reshape_labels_and_masks(loss_mask)
            local_labels = self.morphed_reshape_labels_and_masks(labels)
        else:
            local_loss_mask = None
            local_labels = None

        if self.use_attn_mask_compression:
            attention_mask = self.casual_mask()
        elif attention_mask is None:
            attention_mask = self.casual_mask(input_ids)
        return labels, attention_mask, loss_mask, local_labels, local_loss_mask


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
        self.use_attn_mask_compression = config.use_attn_mask_compression or config.use_eod_attn_mask_compression

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

        # init parallel state groups
        self.tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
        self.dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        self.pp = config.pipeline_model_parallel_size if config.pipeline_model_parallel_size is not None else 1
        self.cp = config.context_parallel_size if config.context_parallel_size is not None else 1

        if _get_parallel_mode() != ParallelMode.STAND_ALONE:
            initialize_model_parallel(tensor_model_parallel_size=self.tp, data_parallel_size=self.dp,
                                      pipeline_model_parallel_size=self.pp, context_parallel_size=self.cp)

        if self.config.track_max_attention_logit:
            self.rank_id = get_rank()
            self.allreduce_max_in_dp = (
                None if self.dp == 1
                else P.AllReduce(op=P.ReduceOp.MAX, group=get_dp_group_name(self.rank_id, self.dp, self.tp, self.cp)[0])
            )
            self.allreduce_max_in_cp = (
                None if self.cp == 1
                else P.AllReduce(op=P.ReduceOp.MAX, group=get_cp_group_name(self.rank_id, self.dp, self.tp, self.cp)[0])
            )

        self.preprocess_labels_and_masks = PreprocessLabelsAndMasks(config)

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
        elif self.position_embedding_type == 'none':
            self.rotary_pos_emb = None
        if config.rotary_dtype == dtype.float16:
            raise ValueError("rotary_dtype `float16` is not supported now.")
        if config.rotary_dtype != dtype.float32:
            logger.warning("For training stability, rotary_dtype is recommended to `float32`.")

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
        self.assign = aclnn_ops.InplaceCopy()

        # init morphed layer
        self.morphed_reshape_logits = Morph(
            self.forward_func_logits,
            func_infer_shape,
            func_infer_dtype
            ).add_prim_attr("self_define_shard", True)

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
        if config.moe_router_load_balancing_type == "gbs_aux_loss":
            self.zeros_tensor = ms.Tensor(np.zeros([config.num_moe_experts]), ms.float32)
            self.assign_aux = aclnn_ops.Assign()

        rules = config.param_init_std_rules
        if rules is not None:
            self.update_weight(rules)

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        else:
            self.shard(config)

    def update_weight(self, rules):
        logger.info(f"update weight with rules {rules}")
        for rule_ in rules:
            pattern = rule_['target']
            init_method_std = rule_['init_method_std']
            for param_name, param in self.parameters_and_names():
                if pattern.match(param_name):
                    weight_shape = list(param.shape)
                    param.set_data(init_method_normal(init_method_std)(weight_shape))
                    logger.info(f"{param_name} initial weight will be update, new init_std: {init_method_std}")

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

        labels, attention_mask, loss_mask, local_labels, local_loss_mask = self._preprocess_input_labels_and_masks(
            input_ids, labels, attention_mask, loss_mask)

        hidden_states, rotary_pos_emb, extra_loss = self.language_model(
            input_ids,
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
                input_ids,
                position_ids,
                hidden_states,
                attention_mask,
                labels=labels,
                rotary_pos_emb=rotary_pos_emb,
                loss_mask=loss_mask,
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
            logits = self.morphed_reshape_logits(logits)
        logits = self.cast(logits, dtype.float32)

        if not self.training:
            return logits.contiguous()

        # labels origin shape is [b s], Transpose is not required.
        loss = self.compute_language_model_loss(local_labels, logits, local_loss_mask)

        if self.calculate_per_token_loss:
            numerator0, denominator0 = loss
            return numerator0, denominator0, numerator1, denominator1, extra_loss * denominator0
        return loss, mtp_loss, extra_loss

    def forward_func_logits(self, input_):
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
        return self.preprocess_labels_and_masks(input_ids, labels, attention_mask, loss_mask)

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

    def reset_accu_gbs_fi(self,):
        num_layers = self.config.num_layers
        for i in range(num_layers):
            if hasattr(self.decoder.layers[i].mlp, "router"):
                self.assign_aux(self.decoder.layers[i].mlp.router.fi_accu, self.zeros_tensor)

    def _iter_core_attentions(self):
        """Iterate over all core_attention modules with their param names.

        Yields:
            Tuple[str, module]: A tuple of (param_name, core_attention_module).
        """
        num_layers = self.config.num_layers
        mtp_num_layers = self.config.mtp_num_layers

        for i in range(num_layers):
            core_attn = self.decoder.layers[i].self_attention.core_attention
            yield f"decoder.layers.{i}.self_attention.core_attention", core_attn

        for i in range(mtp_num_layers):
            core_attn = self.mtp.layers[i].transformer_layer.self_attention.core_attention
            yield f"mtp.layers.{i}.transformer_layer.self_attention.core_attention", core_attn

    def get_max_attention_logit(self):
        """Get max attention logit values for all layers.

        Returns:
            dict: A dictionary mapping parameter names to their max logit values.
                  Only includes layers with valid (sum > 0) max_logits_val.
        """
        max_logits = {}
        for param_name, core_attn in self._iter_core_attentions():
            if not hasattr(core_attn, "max_logits_val"):
                continue
            param = core_attn.max_logits_val.value()
            if param.sum() <= 0:
                continue
            max_logits[f"{param_name}.max_logits_val"] = param
        return max_logits

    def allreduce_max_attention_logit(self):
        """
        Perform AllReduce-Max operation across DP and CP dimensions for max attention logits.

        This method aggregates the maximum attention logit values from all data parallel
        and context parallel ranks to ensure consistent max logit values across the model.
        """
        num_layers = self.config.num_layers
        mtp_num_layers = 0 if self.config.mtp_num_layers is None else self.config.mtp_num_layers

        def _allreduce_max_param(max_logits):
            param = max_logits.value()
            if self.allreduce_max_in_dp is not None:
                param = self.allreduce_max_in_dp(param)
            if self.allreduce_max_in_cp is not None:
                param = self.allreduce_max_in_cp(param)
            self.assign(max_logits, param)

        for i in range(num_layers):
            max_logits = self.decoder.layers[i].self_attention.core_attention.max_logits_val
            _allreduce_max_param(max_logits)

        for i in range(mtp_num_layers):
            max_logits = self.mtp.layers[i].transformer_layer.self_attention.core_attention.max_logits_val
            _allreduce_max_param(max_logits)

    def reset_max_attention_logit(self):
        """Reset max attention logit to zeros for all layers."""
        for _, core_attn in self._iter_core_attentions():
            if hasattr(core_attn, "max_logits_val"):
                param = core_attn.max_logits_val
                F.assign(param, F.zeros_like(param))

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

        if config.moe_router_load_balancing_type == "gbs_aux_loss":
            self.assign_aux.shard(((1,), (1,)))

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
                if self.use_attn_mask_compression:
                    self.preprocess_labels_and_masks.pipeline_stage = 0
                    self.preprocess_labels_and_masks.pipeline_segment = 1
            else:
                self.output_layer.pipeline_stage = pipeline_stage - 1
                if self.mtp_process:
                    self.mtp.pipeline_stage = pipeline_stage - 1
                if self.use_attn_mask_compression:
                    self.preprocess_labels_and_masks.pipeline_stage = pipeline_stage - 1
        self.morphed_reshape_logits.shard(
            in_strategy=(
                layout("dp", "cp", "tp"),
            ),
            out_strategy=(
                layout("dp_cp", "tp"),
            )
        )
        if self.config.track_max_attention_logit:
            if self.allreduce_max_in_dp is not None:
                self.allreduce_max_in_dp.shard((layout("tp"),))
            if self.allreduce_max_in_cp is not None:
                self.allreduce_max_in_cp.shard((layout("tp"),))

    def sharding_propagation(self, config: TransformerConfig):
        pass

    def sharded_state_dict(self):
        """Get all sharded state dict."""
        sharded_state_dict = {}
        for _, sub_cell in self.cells_and_names():
            if sub_cell != self and hasattr(sub_cell, "sharded_state_dict"):
                sharded_state_dict.update(sub_cell.sharded_state_dict())
        return sharded_state_dict

    def get_model_parameters(self):
        """Get current rank trainable parameters in gpt model ."""
        params = set()
        current_pipeline_stage = get_current_rank_stage()
        if ms.get_auto_parallel_context('pipeline_stages') > 1:
            if current_pipeline_stage == self.output_layer.pipeline_stage:
                params.update(get_model_parameters(self.output_layer))
            if hasattr(self, "mtp"):
                if current_pipeline_stage == self.mtp.pipeline_stage:
                    params.update(get_model_parameters(self.mtp))
            params.update(self.decoder.get_model_parameters())
            params.update(get_model_parameters(self.embedding))
        else:
            params.update(get_model_parameters(self))
        return params

    def make_model_muon_fns(self,):
        """Read values from TransformersConfig and generate schema."""

        num_moe_experts = self.config.num_moe_experts
        hidden_size = self.config.hidden_size
        moe_ffn_hidden_size = self.config.moe_ffn_hidden_size
        qk_head_dim = self.config.qk_head_dim
        qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        num_attention_heads = self.config.num_attention_heads
        kv_lora_rank = self.config.kv_lora_rank
        value_head_dim = self.config.v_head_dim

        schema = [
            # experts.weight1: reshape → split into two [num_moe_experts, hidden_size, moe_ffn_hidden_size]
            {
                "patterns": ["*mlp.experts.weight1*"],
                "kind": "reshape_concat",
                "reshape": (num_moe_experts, hidden_size, 2 * moe_ffn_hidden_size),
            },
            # experts.weight2: reshape → [num_moe_experts, moe_ffn_hidden_size, hidden_size]
            {
                "patterns": ["*mlp.experts.weight2*"],
                "kind": "reshape_only",
                "reshape": (num_moe_experts, moe_ffn_hidden_size, hidden_size),
            },
            # q_proj / q_up_proj: periodic split across heads
            {
                "patterns": [
                    "*self_attention.linear_q_proj.weight*",
                    "*self_attention.linear_q_up_proj.weight*",
                ],
                "kind": "periodic",
                "parts": (qk_head_dim, qk_pos_emb_head_dim, num_attention_heads),
            },
            # kv_down_proj: one block
            {
                "patterns": ["*self_attention.linear_kv_down_proj.weight*"],
                "kind": "periodic",
                "parts": (kv_lora_rank, qk_pos_emb_head_dim, 1),
            },
            # kv_up_proj: periodic split across heads
            {
                "patterns": ["*self_attention.linear_kv_up_proj.weight*"],
                "kind": "periodic",
                "parts": (qk_head_dim, value_head_dim, num_attention_heads),
            },
            # fc1 and shared_fc1: alternating 1,1 split along rows
            {
                "patterns": [
                    "*mlp.shared_experts.linear_fc1.weight*",
                    "*mlp.linear_fc1.weight*",
                ],
                "kind": "alt_pair_periodic",
            },
        ]

        return make_muon_fns(schema)

    def get_muon_filter(self):
        """Return a filter function to determine if a parameter should use Muon optimization.
        
        Returns:
            A function that takes a parameter and returns True if it should use Muon.
        """
        def muon_filter(param):
            return (
                (len(param.shape) == 2 or len(param.shape) == 3)
                and "word_embeddings" not in param.name
                and "output_layer" not in param.name
            )
        return muon_filter

    def get_tp_dims(self, params):
        """Return tensor parallel dimensions for each parameter.
        
        Args:
            params: List of parameters from the optimizer.
            
        Returns:
            Tuple of TP dimensions for each parameter.
        """
        no_tp_list = [
            "linear_q_down_proj",
            "linear_kv_down_proj",
            "shared_experts",
            "mlp.router",
            "hnorm.weight", "enorm.weight", "eh_proj.weight",
        ]

        tp_dim_1_list = [
            "self_attention.linear_proj.weight",
            "mlp.linear_fc2.weight"
        ]

        def name_filter(param_name, full_name_list):
            for full_name in full_name_list:
                if full_name in param_name:
                    return True
            return False

        tp_dims = []
        for param in params:
            if name_filter(param.name, tp_dim_1_list):
                tp_dims.append(1)
            elif name_filter(param.name, no_tp_list):
                tp_dims.append(-1)
            else:
                tp_dims.append(0)
        return tuple(tp_dims)

    def get_op_groups_info(self, params, op):
        """Return optimizer parallel group information for each parameter.
        
        Args:
            params: List of parameters from the optimizer.
            op: Optimizer parallel size.
            tp_group: Tensor parallel group name.
            
        Returns:
            Tuple of (ops, op_groups) where:
                - ops: tuple of op values for each parameter
                - op_groups: tuple of group names for each parameter
        """
        no_op_list = [
            "self_attention.linear_q_proj.weight",
            "self_attention.linear_q_up_proj.weight",
            "self_attention.linear_q_down_proj.weight",
            "self_attention.linear_kv_up_proj.weight",
            "self_attention.linear_kv_down_proj.weight",
            "eh_proj",
            "max_logits_val"
        ]

        sharded_state_dict = self.sharded_state_dict()
        world_size = get_group_size()
        ep = self.config.expert_model_parallel_size
        pp = self.config.pipeline_model_parallel_size

        def name_filter(param_name, full_name_list):
            for full_name in full_name_list:
                if full_name in param_name:
                    return True
            return False

        op_list = []
        op_groups = []

        for param in params:
            if name_filter(param.name, no_op_list):
                op_list.append(1)

                op_groups.append("")
                if param.parallel_optimizer:
                    param.parallel_optimizer = False
                    logger.warning(
                        f"Parameter {param.name}: parallel_optimizer was set to False due to the use of Muon optimizer."
                    )
                continue

            # compute real op size
            sharded_info = sharded_state_dict.get(param.name)
            real_op_size, weight_sharded_size = compute_repeat_num_and_model_parallel_size(sharded_info, world_size, pp,
                                                                                           op)
            if real_op_size == 1:
                op_list.append(1)
                op_groups.append("")
                logger.info(f"Parameter {param.name} : No op group.")
                continue

            op_list.append(real_op_size)
            op_group_name, rank_list = get_op_group_name(get_rank(), real_op_size, weight_sharded_size)
            logger.info(f"Parameter {param.name} : Muon real_op_size={real_op_size} group list is: {rank_list}")
            op_groups.append(op_group_name)

        # check if op is valid for expert
        for param, real_op_size in zip(params, op_list):
            if "mlp.experts.weight" not in param.name:
                continue
            # Validate MoE expert counts divisibility constraint:
            # num_moe_experts must be divisible by (optimizer_weight_shard_size * expert_model_parallel_size)
            num_moe_experts = self.config.num_moe_experts
            if bool(num_moe_experts and num_moe_experts > 0):
                if num_moe_experts % (real_op_size * ep) != 0:
                    error_msg =  (f"Invalid configuration: 'num_moe_experts' ({num_moe_experts}) must be divisible by "
                        f"'real_op_size * expert_model_parallel_size' ({real_op_size} * "
                        f"{ep} = {real_op_size * ep}).\n"
                        f"Hint:\n"
                        f"    Although you set `optimizer_weight_shard_size={op}`, the maximum optimizer shard size "
                        f"for `{param.name}` is `{real_op_size}`. Try reducing 'optimizer_weight_shard_size'.")
                    logger.error(error_msg)
                    raise ValueError(
                        error_msg
                    )

        return tuple(op_list), tuple(op_groups)

    def get_param_layer_indices(self, params):
        """Return layer indices for each parameter (used for QK-clip).

        Args:
            params: List of parameters from the optimizer.
            
        Returns:
            Tuple of layer indices for each parameter, where:
                - layer_idx >= 0 stands for the layer_idx-th decoder layer
                - layer_idx < 0 stands for the -(layer_idx+1)-th MTP layer
        """
        param_layer = []
        for param in params:
            name = param.name
            try:
                layer_idx = int(name.split(".")[2])
            except (ValueError, IndexError):
                layer_idx = 0
            if name.startswith('mtp'):
                layer_idx = -layer_idx - 1
            param_layer.append(layer_idx)
        return tuple(param_layer)

    def apply_qk_clip_scaling(self, params, param_names, param_layer, logit_threshold,
                               muon_split_fn, muon_merge_fn):
        """Apply QK-clip scaling to attention weight parameters.

        Args:
            params: List of all parameters.
            param_names: Tuple of parameter names.
            param_layer: Tuple of layer indices for each parameter.
            logit_threshold: Threshold for logit clipping.
            muon_split_fn: Function to split parameters.
            muon_merge_fn: Function to merge parameters.
            
        Returns:
            List of (param_idx, scaled_weights) tuples to be updated.
        """
        self.allreduce_max_attention_logit()
        if not self.config.multi_latent_attention:
            return []
        ones = ms.Tensor([1.0], dtype.float32)
        qk_head_dim = self.config.qk_head_dim
        qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim

        def get_scale_broadcast(scales, head_dim):
            scale_broadcast = ops.tile(ops.expand_dims(scales, 1), (1, head_dim)).reshape(-1)
            scale_broadcast = ops.expand_dims(scale_broadcast, 1)
            return scale_broadcast

        updates = []
        for idx, param_name in enumerate(param_names):
            if (
                "self_attention.linear_q_proj.weight" not in param_name
                and "self_attention.linear_q_up_proj.weight" not in param_name
                and "self_attention.linear_kv_up_proj.weight" not in param_name
            ):
                continue

            layer_idx = param_layer[idx]
            param = params[idx]

            # Compute per-head scale factor
            logit_threshold_f32 = ops.cast(logit_threshold, dtype=dtype.float32)
            if layer_idx >= 0:
                logits_row = (
                    self.decoder.layers[layer_idx]
                    .self_attention
                    .core_attention
                    .max_logits_val
                    .value()
                )
            else:
                logits_row = (
                    self.mtp.layers[-(layer_idx + 1)]
                    .transformer_layer
                    .self_attention
                    .core_attention
                    .max_logits_val
                    .value()
                )

            logits_row = logits_row.reshape(-1)
            mask = ops.greater_equal(logits_row, logit_threshold_f32)
            safe_den = ops.where(mask, logits_row, ones)
            scales = ops.where(mask, logit_threshold_f32 / safe_den, ones)

            weights = None
            if (
                "self_attention.linear_q_proj.weight" in param_name
                or "self_attention.linear_q_up_proj.weight" in param_name
            ):
                l2q_nope_proj, l2q_pe_proj = muon_split_fn(param_name, param)
                l2q_nope_proj *= get_scale_broadcast(ops.sqrt(scales), qk_head_dim)
                l2q_pe_proj *= get_scale_broadcast(scales, qk_pos_emb_head_dim)
                weights = muon_merge_fn(param_name, [l2q_nope_proj, l2q_pe_proj])
            elif "self_attention.linear_kv_up_proj.weight" in param_name:
                lkv2kv_k_nope, lkv2kv_v = muon_split_fn(param_name, param)
                lkv2kv_k_nope *= get_scale_broadcast(ops.sqrt(scales), qk_head_dim)
                weights = muon_merge_fn(param_name, [lkv2kv_k_nope, lkv2kv_v])

            if weights is not None:
                updates.append((idx, weights))

        return updates
