# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Telechat models' APIs."""
import copy
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.ops.functional as F
from mindspore import Tensor, nn, mint
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from research.telechat3.telechat_transformer import TelechatDecodeLayer
from research.telechat3.telechat_interleave import TelechatDecodeLayerInterleave
from research.telechat3.telechat_layer import TelechatEmbedding
from research.telechat3.telechat_config import TelechatConfig

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.utils import LayerSetting, lazy_inline, check_fine_grain_interleave_valid
from mindformers.models.llama.llama_layer import LlamaRMSNorm
from mindformers.modules.layers import Linear, FreqsMgr
from mindformers.modules.transformer import LowerTriangularMaskWithDynamic
from mindformers.modules.transformer.op_parallel_config import _check_config, default_dpmp_config
from mindformers.tools.logger import logger
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_predict_run_mode


class TelechatPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TelechatConfig
    base_model_prefix = "telechat"


class AttentionMaskWithEod(nn.Cell):
    """
    This class generates a two-dimensional attention mask matrix based on the input token sequence and
    the end-of-document marker. It not only implements standard causal/padding masking but also ensures
    attention isolation between different documents.
    """
    def __init__(self, seq_length, parallel_config=default_dpmp_config, compute_dtype=mstype.float16,
                 use_flash_attention=False, **kwargs):
        super().__init__()
        dp = parallel_config.data_parallel
        self.seq_length = seq_length
        self.compute_dtype = compute_dtype
        self.use_flash_attention = use_flash_attention
        self.one = Tensor([1.0], dtype=compute_dtype)
        self.multiply_data = Tensor([-10000.0], dtype=compute_dtype)
        #self.arange_tensor = Tensor(np.tile(np.arange(seq_length).reshape(1, 1, seq_length), (1, seq_length, 1)),
        #                            mstype.int16)
        self.less = P.LessEqual().shard(((dp, 1, 1), (1, 1, 1)))
        self.tile = P.Tile().shard(((dp, 1, 1),))
        self.expand_dims = P.ExpandDims().shard(((dp, 1),))
        self.mul = P.Mul().shard(((1, 1, 1), (dp, 1, 1)))
        self.sub = P.Sub().shard(((1,), (dp, 1, 1)))
        self.cast = P.Cast()
        self.mul_post = P.Mul().shard(((dp, 1, 1, 1), (1,)))
        self.expand_dim_post = P.ExpandDims().shard(((dp, 1, 1),))
        self.range = P.Range().shard(((1,),))
        self.less_equal = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
        self.reshape = P.Reshape()

    def construct(self, sequence_start_ids):
        """
        Forward of telechat model.

        Args:
            sequence_start_ids: the start_ids of tokenized inputs with datatype int32
        Returns:
            output: Tensor
        """
        col_indices = self.range(0, self.seq_length, 1)
        col_indices = self.reshape(col_indices, (1, 1, self.seq_length))

        row_indices = self.reshape(col_indices, (1, self.seq_length, 1))

        lower_triangle_mask = self.less_equal(col_indices, row_indices)

        start_pos = self.cast(self.expand_dims(sequence_start_ids, -1), mstype.int32)
        doc_mask = self.less(start_pos, col_indices)

        combined_mask = self.mul(self.cast(lower_triangle_mask, self.compute_dtype),
                                 self.cast(doc_mask, self.compute_dtype))
        return combined_mask

    def post_process(self, mask):
        mask = self.sub(self.one, self.cast(mask, self.compute_dtype))
        if not self.use_flash_attention:
            mask = self.expand_dim_post(mask, 1)
            mask = self.mul_post(mask, self.multiply_data)
        else:
            mask = self.expand_dim_post(mask, 1)
            mask = self.cast(mask, mstype.uint8)
        return mask


class TelechatModel(TelechatPreTrainedModel):
    r"""
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TelechatDecoderLayer`]
    Args:
        config(TelechatConfig): the config of network

    Returns:
            output: Tensor, the output of telechat decoderlayer

    Examples:
        >>> from mindformers import TelechatModel
        >>> network = TelechatModel.from_pretrained('telechat_36b')
        >>> type(network)
        <class 'mindformers.models.telechat.telechat.TelechatModel'>
    """
    def __init__(self, config: TelechatConfig=None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.model_name=config.net_name
        self.dtype=config.compute_dtype
        self.hidden_size=config.hidden_size
        self.embedding_size=config.embedding_size
        self.num_layers=config.num_layers
        self.n_head=config.num_heads
        self.head_dim=self.hidden_size // self.n_head
        self.pad_token_id=config.pad_token_id
        self.is_first_iteration=True
        self.use_past=config.use_past
        self.use_flash_attention=config.use_flash_attention

        self.concat = P.Concat(-1)
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()

        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=config.seq_length,
                                  max_position_embedding=config.max_position_embedding,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method,
                                  parallel_config=config.parallel_config,
                                  is_dynamic=config.is_dynamic)
        if config.eod_reset:
            self.casual_mask = AttentionMaskWithEod(seq_length=config.seq_length,
                                                    parallel_config=config.parallel_config,
                                                    compute_dtype=config.compute_dtype,
                                                    use_flash_attention=config.use_flash_attention)
        else:
            self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.seq_length,
                                                             compute_type=config.compute_dtype,
                                                             is_dynamic=config.is_dynamic,
                                                             pad_token_id=config.pad_token_id,
                                                             use_flash_attention=config.use_flash_attention,
                                                             use_attn_mask_compression=config.use_attn_mask_compression,
                                                             use_past=config.use_past)
        self.tok_embeddings = TelechatEmbedding(vocab_table_size=config.vocab_size,
                                                sigma=config.sigma,
                                                mean=config.mean,
                                                embedding_size=config.hidden_size,
                                                param_init_type=config.embedding_init_type,
                                                parallel_optimizer=config.parallel_optimizer)
        self.norm_out = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps,
                                        compute_type=config.layernorm_compute_type)
        self.fine_grain_interleave = check_fine_grain_interleave_valid(config.fine_grain_interleave,
                                                                       config.parallel_config)
        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)
        for layer_id in range(config.num_layers):
            if self.fine_grain_interleave:
                layer = TelechatDecodeLayerInterleave(config.seq_length,
                                                      layer_id,
                                                      dim=config.hidden_size,
                                                      n_heads=config.num_heads,
                                                      num_layers=config.num_layers,
                                                      n_kv_heads=config.n_kv_heads,
                                                      intermediate_size=config.intermediate_size,
                                                      ffn_dim_multiplier=config.ffn_dim_multiplier,
                                                      norm_eps=config.rms_norm_eps,
                                                      qkv_has_bias=config.qkv_has_bias,
                                                      out_proj_has_bias=config.out_proj_has_bias,
                                                      compute_dtype=config.compute_dtype,
                                                      layernorm_compute_dtype=config.layernorm_compute_type,
                                                      softmax_compute_dtype=config.softmax_compute_type,
                                                      rotary_dtype=config.rotary_dtype,
                                                      param_init_type=config.param_init_type,
                                                      res_dtype=config.res_dtype,
                                                      use_flash_attention=config.use_flash_attention,
                                                      use_attn_mask_compression=config.use_attn_mask_compression,
                                                      use_rope_slice=config.use_rope_slice,
                                                      fine_grain_interleave=config.fine_grain_interleave,
                                                      parallel_config=config.parallel_config)
            else:
                layer = TelechatDecodeLayer(layer_id,
                                            dim=config.hidden_size,
                                            n_heads=config.num_heads,
                                            n_kv_heads=config.n_kv_heads,
                                            sigma=config.sigma,
                                            mean=config.mean,
                                            moe_config=config.moe_config,
                                            intermediate_size=config.intermediate_size,
                                            multiple_of=config.multiple_of,
                                            ffn_dim_multiplier=config.ffn_dim_multiplier,
                                            norm_eps=config.rms_norm_eps,
                                            qkv_has_bias=config.qkv_has_bias,
                                            out_proj_has_bias=config.out_proj_has_bias,
                                            qkv_concat=config.qkv_concat,
                                            compute_dtype=config.compute_dtype,
                                            layernorm_compute_dtype=config.layernorm_compute_type,
                                            softmax_compute_dtype=config.softmax_compute_type,
                                            rotary_dtype=config.rotary_dtype,
                                            param_init_type=config.param_init_type,
                                            res_dtype=config.res_dtype,
                                            use_past=config.use_past,
                                            use_flash_attention=config.use_flash_attention,
                                            use_attn_mask_compression=config.use_attn_mask_compression,
                                            block_size=config.block_size,
                                            num_blocks=config.num_blocks,
                                            is_dynamic=config.is_dynamic,
                                            use_rope_slice=config.use_rope_slice,
                                            parallel_config=config.parallel_config)
            self.layers.append(layer)
        dp = config.parallel_config.data_parallel
        sp = config.parallel_config.context_parallel
        self.sp = sp
        self.expert_num = 1 if config.moe_config is None else config.moe_config.expert_num
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.tok_embeddings.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.norm_out.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.tok_embeddings.set_comm_fusion(2)
                self.norm_out.set_comm_fusion(2)
            else:
                self.tok_embeddings.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
                self.norm_out.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

            self.tok_embeddings.shard(config.parallel_config)
            #self.casual_mask.shard(config.parallel_config)
            self.concat.shard(((dp, 1, 1, 1), (dp, 1, 1, 1)))
            if self.fine_grain_interleave:
                self.norm_out.shard((dp * sp, 1))
            else:
                self.norm_out.shard((dp, sp, 1))

    # pylint: disable=W0613
    def construct(self, tokens: Tensor, batch_valid_length=None, batch_index=None, zactivate_len=None, aux_loss=None,
                  block_tables=None, attention_mask=None, slot_mapping=None, prefix_keys_values=None):
        """
        Forward of telechat model.

        Args:
            tokens: the tokenized inputs with datatype int32
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.
        Returns:
            output: Tensor, the output of telechat decoderlayer
        """
        # preprocess
        bs, seq_len = self.shape(tokens)
        mask = attention_mask
        if self.use_past:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill(bs, seq_len)
                mask = self.casual_mask.prefill()
                if prefix_keys_values is not None:
                    if mask is None:
                        mask = self.casual_mask(tokens)
                    prefix_length = prefix_keys_values[0].shape[2]
                    prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                    mask = self.concat((prefix_mask, mask))
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)
        else:
            if attention_mask is not None:
                if not self.config.eod_reset:
                    mask = attention_mask
                    mask = self.cast(mask, mstype.uint8)
                    freqs_cis = self.freqs_mgr(seq_len)
                else:
                    mask = self.casual_mask(mask)
                    mask = self.casual_mask.post_process(mask)
                    freqs_cis = self.freqs_mgr(seq_len)
            else:
                mask = self.casual_mask(tokens)
                freqs_cis = self.freqs_mgr(seq_len)
                if prefix_keys_values is not None:
                    prefix_length = prefix_keys_values[0].shape[2]
                    prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                    mask = self.concat((prefix_mask, mask))

        # tokens: [bs, seq/1]
        h = self.tok_embeddings(tokens)
        h = self.reshape(h, (bs, seq_len, -1))
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            prefix_kv = prefix_keys_values[i] if prefix_keys_values is not None else None
            if self.expert_num > 1:
                h, aux_loss = self.layers[i](h, freqs_cis, mask, batch_valid_length=batch_valid_length,
                                             block_tables=block_tables, aux_loss=aux_loss, slot_mapping=slot_mapping,
                                             prefix_keys_values=prefix_kv)
            else:
                h = self.layers[i](h, freqs_cis, mask, batch_valid_length=batch_valid_length, block_tables=block_tables,
                                   slot_mapping=slot_mapping, prefix_keys_values=prefix_kv)
        output = self.norm_out(h)
        return output


class TelechatHead(nn.Cell):
    """Head for Telechat to get the logits of each token in the vocab."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 compute_dtype="float16",
                 parallel_config=None):
        super().__init__()
        copied_parallel_config = copy.deepcopy(parallel_config)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = compute_dtype
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        dp = copied_parallel_config.data_parallel
        mp = copied_parallel_config.model_parallel
        sp = copied_parallel_config.context_parallel
        if parallel_config.vocab_emb_dp or (out_channels % mp != 0):
            self.matmul = P.MatMul(transpose_b=True).shard(((dp * sp, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((dp * sp, 1), (mp, 1)))

    def construct(self, x, embedding_weight=None):
        out_shape = P.Shape()(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(embedding_weight, self.dtype)
        x = self.cast(x, self.dtype)
        x = self.matmul(x, weight)
        x = self.cast(x, ori_dtype)
        output = self.reshape(x, out_shape)
        return output


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class TelechatForCausalLM(TelechatPreTrainedModel):
    """
    Provide telechat training loss or logits through network.

    Args:
        config (TelechatConfig): The config of telechat model.

    Returns:
        output: Tensor, the output of telechat decoderlayer

    Examples:
        >>> from mindformers.models.telechat import TelechatConfig, TelechatForCausalLM
        >>> config = TelechatConfig(batch_size=2)
        >>> network = TelechatForCausalLM(config=config)
        >>> type(network)
        <class 'mindformers.models.telechat.telechat.TelechatForCausalLM'>
        >>> from mindformers import TelechatForCausalLM
        >>> network = TelechatForCausalLM.from_pretrained('telechat_115b')
        >>> type(network)
        <class 'mindformers.models.telechat.telechat.TelechatForCausalLM'>
    """

    @lazy_inline
    def __init__(self, config: TelechatConfig = None):
        super().__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.config = config
        self.model_name = config.net_name
        self.ignore_token_id = config.ignore_token_id
        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.vocab_size = config.vocab_size
        self.rl_config = config.rl_config
        self.is_first_iteration = True

        self.dp = config.parallel_config.data_parallel
        self.mp = config.parallel_config.model_parallel
        self.expert_num = config.moe_config.expert_num
        self.init_aux_loss = Tensor(np.zeros([self.dp * self.mp, self.expert_num]), mstype.float32)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather()
        self.sub_batch_valid_len = P.Sub()
        self.model = TelechatModel(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                                out_channels=config.vocab_size,
                                has_bias=False,
                                compute_dtype=config.compute_dtype,
                                param_init_type=config.param_init_type,
                                weight_init="normal")  # meta default: xavier_normal

        mp = config.parallel_config.model_parallel
        sp = config.parallel_config.context_parallel
        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1
        loss_parallel_config.data_parallel *= loss_parallel_config.context_parallel
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad)

        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.aux_reduce_mean = P.ReduceMean(keep_dims=True).shard(((1, 1),))
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.slice.shard(((dp, 1),))
            self.not_equal.shard(((dp, 1), ()))
            self.mul.shard(((dp, 1), (dp, 1)))
            self.add.shard(((dp, 1), ()))
            self.gather.shard(((dp, 1, 1), (dp,)))
            self.sub_batch_valid_len.shard(((1,), ()))
            if config.parallel_config.vocab_emb_dp or (vocab_size % mp != 0):
                self.lm_head.shard(strategy_matmul=((dp * sp, 1), (1, 1)))
            else:
                self.lm_head.shard(strategy_matmul=((dp * sp, 1), (mp, 1)))
            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.predict_run_mode = get_predict_run_mode()

        logger.info(f"Predict run mode:{self.predict_run_mode}")

    # pylint: disable=W0613
    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """Get Telechat model input tuple for transform ckpt."""
        input_ids = Tensor(input_ids, mstype.int32)
        labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
        bs, seq = input_ids.shape[0], input_ids.shape[1]
        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        prefix_keys_values = Tensor(kwargs["prefix_keys_values"]) if "prefix_keys_values" in kwargs else None
        return input_ids, labels, None, None, None, None, None, None, None, None, None, None, \
               slot_mapping, prefix_keys_values

    def set_dynamic_inputs(self, **kwargs):
        """Set dynamic inputs"""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mstype.float16)
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, dynamic_prefix_keys_values)
        else:
            self.set_inputs(dynamic_input_ids, None, None, None, None, None, None, None,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, None)
        logger.info("Set dynamic input for telechat.")

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.attention.infer_attention.add_flags(is_first_iteration=is_first_iteration)

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, loss_mask=None, input_position=None, position_ids=None,
                  attention_mask=None, input_embeds=None, init_reset=None, batch_valid_length=None,
                  batch_index=None, zactivate_len=None, block_tables=None, slot_mapping=None,
                  prefix_keys_values=None):
        r"""
        TelechatForCausalLM forward.

        Args:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): Reserved param, not used.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.
        Returns:
            Tensor: The loss or (logits, tokens, input_mask) of the network.
        """
        bsz, seqlen = self.shape(input_ids)
        aux_loss = None
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        if self.expert_num == 1:
            output = self.model(tokens, batch_valid_length, batch_index, zactivate_len, block_tables=block_tables, \
                                attention_mask=attention_mask, slot_mapping=slot_mapping, \
                                prefix_keys_values=prefix_keys_values)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            batch_valid_length = mint.cumsum(batch_valid_length, 0)
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        if self.rl_config is not None:
            return logits

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        if not self.training:
            logits = self.cast(logits, mstype.float32)
            if self.predict_run_mode:
                logits = self.reshape(logits, (-1, logits.shape[-1]))
                return logits
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        if self.expert_num > 1:
            aux_loss = self.aux_reduce_mean(aux_loss).reshape(-1)
            loss = loss + aux_loss
        return loss

    def kvcache(self, layer_idx):
        key_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache
