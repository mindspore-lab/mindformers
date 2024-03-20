# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Qwen models' APIs."""

import copy
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import log as logger
from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
try:
    # pylint: disable=W0611
    from mindspore.nn.layer.flash_attention import FlashAttention

    FLASHATTENTION_VALID = True
except ImportError:
    FLASHATTENTION_VALID = False

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.models.base_model import BaseModel
from mindformers.models.utils import cell_reuse
from mindformers.tools.logger import _LogActionOnce
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.modules.layers import Linear, _check_input_dtype, _args_type_validator_check, _valid_value_checks
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.models.llama.llama import LlamaForCausalLM, layer_compute_dtype
from mindformers.models.llama.llama_layer import LlamaEmbedding, FreqsMgr, LlamaSiLU, LlamaRMSNorm
from mindformers.models.llama.llama_transformer import LLamaDecodeLayer
from mindformers.modules import KVCachePreprocess
from mindformers.version_control import check_valid_paged_attention


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class QwenForCausalLM(BaseModel):
    r"""
        Provide qwen training loss or logits through network.
        Args:
            config (QwenConfig): The config of Qwen model.

        Returns:
            Tensor, the loss or logits of the network.
        """

    @cell_reuse
    def __init__(self, config=None):
        super().__init__(config)

        self.transformer = QwenModel(config=config)
        self.lm_head = Linear(in_channels=config.hidden_size,
                              out_channels=config.vocab_size,
                              has_bias=False,
                              compute_dtype=config.compute_dtype,
                              param_init_type=config.param_init_type,
                              skip_redistribution=config.is_dynamic,
                              weight_init="normal")
        loss_parallel_config = copy.deepcopy(config.parallel_config)
        loss_parallel_config.model_parallel = loss_parallel_config.model_parallel * loss_parallel_config.data_parallel
        loss_parallel_config.data_parallel = 1
        self.loss = CrossEntropyLoss(parallel_config=loss_parallel_config)

        self.pad_token_id = config.pad_token_id
        self.use_past = config.use_past
        self.ignore_token_id = config.ignore_token_id
        self.seq_length = config.seq_length
        self.vocab_size = config.vocab_size
        self.is_first_iteration = True
        self.not_equal = P.NotEqual()
        self.cast = P.Cast()
        self.add = P.Add()
        self.reshape = P.Reshape()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.ones = P.Ones()
        self.slice = P.StridedSlice()
        self.mul = P.Mul()
        self.sub_batch_valid_len = P.Sub()
        self.gather = P.Gather(1)

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.shard(config.parallel_config)
            if config.parallel_config.pipeline_stage > 1:
                self.lm_head.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.use_paged_attention = self.config.use_paged_attention and check_valid_paged_attention()

        self.load_checkpoint(config)

    # pylint: disable=W0613
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": Tensor(input_ids, mstype.int32)
        }

    def prepare_inputs_for_export(self, full_model=True):
        """Prepare inputs for exported mslite model."""
        use_paged_attention = self.config.use_paged_attention and check_valid_paged_attention()

        if full_model:
            logger.warning("\nExport with settings:" +
                           f"\n  seq_length = {self.seq_length}" +
                           f"\n  batch_size = {self.config.batch_size}" +
                           f"\n  paged_attention = {use_paged_attention}" +
                           (f"\n    pa_block_size = {self.config.block_size}" if use_paged_attention else "") +
                           (f"\n    pa_num_blocks = {self.config.num_blocks}" if use_paged_attention else "") +
                           ("\n  is_dynamic = True" if self.config.is_dynamic else ""))

        return LlamaForCausalLM.prepare_inputs_for_export(self, full_model)

    # pylint: disable=W0613
    def construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                  input_embeds=None, init_reset=True, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        """construct"""
        bsz, seqlen = input_ids.shape
        if self.use_past:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mstype.int32)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = input_ids

        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))
        if not self.is_first_iteration:
            batch_valid_length = self.sub_batch_valid_len(batch_valid_length, 1)

        if self.use_paged_attention and (slot_mapping is None):
            slot_mapping = self.ones((bsz * seqlen,), mstype.int32)

        output = self.transformer(tokens, init_reset=init_reset, batch_valid_length=batch_valid_length,
                                  batch_index=batch_index, zactivate_len=zactivate_len,
                                  block_tables=block_tables, slot_mapping=slot_mapping)
        pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
        if pre_gather:
            output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        logits = self.lm_head(output)

        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), mstype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bsz, seqlen), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mstype.float32)
                input_mask = self.mul(input_mask, label_mask)

        if not self.training:
            if not pre_gather:
                logits = self.reshape(logits, (bsz, seqlen, -1))
            logits = self.cast(logits, mstype.float32)
            # makes cast effective to avoid allgather issue in Mindspore1.10
            input_mask = self.add(input_mask, 1)
            return logits, tokens, input_mask

        if logits.ndim > 2:
            logits = self.reshape(logits, (-1, logits.shape[-1]))
        logits = self.cast(logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))
        loss = self.loss(logits, labels, input_mask)
        return loss

    def shard(self, parallel_config):
        """sharding for feedforward"""

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.slice.shard(((dp, 1),))
        self.not_equal.shard(((dp, 1), ()))
        self.mul.shard(((dp, 1), (dp, 1)))
        self.add.shard(((dp, 1), ()))
        self.sub_batch_valid_len.shard(((1,), ()))
        self.gather.shard(((dp, 1, 1), (dp,)))

        if parallel_config.vocab_emb_dp:
            self.lm_head.shard(strategy_matmul=((dp, 1), (1, 1)))
        else:
            self.lm_head.shard(strategy_matmul=((1, 1), (dp * mp, 1)))


class QwenModel(BaseModel):
    """transformer"""

    def __init__(self, config):
        super().__init__(config)
        self.dtype = config.compute_dtype
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_layers
        self.embed_dim = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.seq_length = config.seq_length
        self.pad_token_id = config.pad_token_id
        self.num_attention_heads = config.num_heads
        self.compute_in_2d = config.compute_in_2d
        self.use_past = config.use_past
        self.is_dynamic = config.is_dynamic
        self.use_kvcache_op = config.use_kvcache_op
        self.is_flexible_shape = config.is_flexible_shape

        self.is_first_iteration = True
        self.use_flash_attention = config.use_flash_attention and FLASHATTENTION_VALID
        if self.use_flash_attention:
            logger.info("Enable flash attention.")
        elif config.use_flash_attention:
            logger.info("Current MindSpore do not support flash attention.")

        self.use_paged_attention = config.use_paged_attention and check_valid_paged_attention()
        if self.use_paged_attention:
            logger.info("Enable paged attention.")

        # 1. wte
        self.wte = LlamaEmbedding(self.vocab_size, self.embed_dim, param_init_type=config.param_init_type,
                                  parallel_optimizer=True)

        # 2. drop
        self.drop = nn.Dropout(p=config.emb_dropout_prob)

        # 4. h hidden layers for transformer
        self.layers = nn.CellList()
        for layer_id in range(config.num_layers):
            layer = QwenDecodeLayer(config.batch_size,
                                    config.seq_length,
                                    layer_id,
                                    dim=config.hidden_size,
                                    n_heads=config.num_heads,
                                    intermediate_size=config.intermediate_size,
                                    norm_eps=config.rms_norm_eps,
                                    compute_dtype=config.compute_dtype,
                                    layernorm_compute_dtype=config.layernorm_compute_type,
                                    softmax_compute_dtype=config.softmax_compute_type,
                                    rotary_dtype=config.rotary_dtype,
                                    param_init_type=config.param_init_type,
                                    qkv_has_bias=True,
                                    use_past=config.use_past,
                                    is_dynamic=self.is_dynamic,
                                    use_kvcache_op=config.use_kvcache_op,
                                    use_flash_attention=self.use_flash_attention,
                                    use_paged_attention=self.use_paged_attention,
                                    block_size=config.block_size,
                                    num_blocks=config.num_blocks,
                                    parallel_config=config.parallel_config)

            layer_compute_dtype(layer, layer_id, config.offset,
                                config.parallel_config, config.num_layers)

            self.layers.append(layer)

        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=self.seq_length,
                                  max_position_embedding=config.max_position_embedding,
                                  rotary_dtype=config.rotary_dtype,
                                  theta=config.theta,
                                  scaling_factor=config.scaling_factor,
                                  extend_method=config.extend_method,
                                  is_dynamic=config.is_dynamic)
        self.casual_mask = CausalMaskForQwen(seq_length=config.seq_length,
                                             compute_type=config.compute_dtype,
                                             is_dynamic=config.is_dynamic,
                                             pad_token_id=config.pad_token_id,
                                             use_flash_attention=config.use_flash_attention)
        self.kvcache_preprocess = KVCachePreprocess(max_batch_size=config.batch_size,
                                                    max_seq_length=config.seq_length,
                                                    is_dynamic=config.is_dynamic,
                                                    use_kvcache_op=config.use_kvcache_op,
                                                    is_flexible_shape=config.is_flexible_shape,
                                                    use_paged_attention=self.use_paged_attention,)
        # 5. ln_f
        self.ln_f = LlamaRMSNorm(self.embed_dim,
                                 eps=config.rms_norm_eps,
                                 compute_type=config.layernorm_compute_type,
                                 is_dynamic=config.is_dynamic,)

        self.shape = P.Shape()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.ones = P.Ones()

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.shard(config.parallel_config)

            self.wte.pipeline_stage = 0
            if config.parallel_config.pipeline_stage > 1:
                self.ln_f.pipeline_stage = config.parallel_config.pipeline_stage - 1
                self.wte.set_comm_fusion(2)
                self.ln_f.set_comm_fusion(2)
            else:
                self.wte.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
                self.ln_f.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

    # pylint: disable=W0613
    def construct(self, input_ids: Tensor, init_reset=True, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None):
        """construct"""
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])

        bs, seq_len = self.shape(input_ids)

        # 1. wte
        h = self.wte(input_ids)
        h = self.reshape(h, (bs, seq_len, self.embed_dim))

        # 2. drop
        hidden_states = self.drop(h)

        # 3. causal mask for attentions
        if not self.use_past:
            freqs_cis = self.freqs_mgr()
            mask = self.casual_mask(input_ids)  # mask: [bs, seq, seq]
            mask = self.casual_mask.post_process(mask)
            kvcache_inputs = None
        else:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr(seq_len)
                mask = self.casual_mask(input_ids)  # mask: [bs, seq, seq]
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length, bs)
                if self.is_dynamic and self.is_flexible_shape and not self.use_kvcache_op:
                    mask = self.casual_mask.increment_slice(self.kvcache_preprocess.range,
                                                            self.kvcache_preprocess.max_cache_length // bs,
                                                            batch_valid_length,
                                                            zactivate_len)
                else:
                    mask = self.casual_mask.increment(self.kvcache_preprocess.range, batch_valid_length, zactivate_len)
            mask = self.casual_mask.post_process(mask)

            kvcache_inputs = self.kvcache_preprocess(bs, batch_valid_length, batch_index, zactivate_len,
                                                     block_tables, slot_mapping)

        # 4. hidden_states
        for i in range(self.num_hidden_layers):
            hidden_states = self.layers[i](hidden_states, freqs_cis, mask, kvcache_inputs=kvcache_inputs)

        # 5. ln_f
        hidden_states = self.ln_f(hidden_states)

        return hidden_states

    def shard(self, parallel_config):
        """sharding for feedforward"""
        self.wte.shard(parallel_config)
        self.casual_mask.shard(parallel_config)
        self.ln_f.shard((parallel_config.data_parallel, 1, 1))


class QwenDecodeLayer(LLamaDecodeLayer):
    """Qwen decode layer"""

    def __init__(self,
                 batch_size,
                 seq_length,
                 layer_id,
                 **kwargs):
        kwargs['qkv_has_bias'] = True
        intermediate_size = kwargs.pop('intermediate_size', 0)
        super().__init__(batch_size, seq_length, layer_id, **kwargs)

        compute_dtype = kwargs.get('compute_dtype', mstype.float16)
        param_init_type = kwargs.get('param_init_type', mstype.float32)
        parallel_config = kwargs.get('parallel_config', TransformerOpParallelConfig())

        is_dynamic = kwargs.get('is_dynamic', False)
        self.feed_forward = QwenFeedForward(dim=self.hidden_size,
                                            intermediate_size=intermediate_size,
                                            compute_dtype=compute_dtype,
                                            param_init_type=param_init_type,
                                            is_dynamic=is_dynamic)

        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.feed_forward.shard(parallel_config)
            self.feed_forward.mul.shard(((dp, 1, mp), (dp, 1, mp)))
        if parallel_config.use_seq_parallel and self.is_first_iteration:
            self.feed_forward.w2.shard(((dp, mp), (1, mp)), out_strategy_matmul=((dp * mp, 1),))

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.attention.wq.bias_add.shard(((dp, mp), (mp,)))
            self.attention.wk.bias_add.shard(((dp, mp), (mp,)))
            self.attention.wv.bias_add.shard(((dp, mp), (mp,)))


class QwenFeedForward(nn.Cell):
    r"""
    Qwen FeedForward.

    .. math::
            (xW_1 * xW_3)W_2

        Inputs:
            - **x** (Tensor) - should be `[batch, seq_length, hidden_size] or [batch * seq_length, hidden_size]`.
              Float tensor.

        Outputs:
            Tensor, the output of this layer after mapping. The shape is `[batch, seq_length, hidden_size] or
            [batch * seq_length, hidden_size]`.

        Raises:
            ValueError: `hidden_dim` is not a multiple of the model parallel way.
            ValueError: `dim` is not a multiple of the model parallel way.
    """

    @_LogActionOnce(m_logger=logger, key='FeedForward',
                    no_warning=_get_parallel_mode() in (ParallelMode.STAND_ALONE,))
    @_args_type_validator_check(dim=Validator.check_positive_int,
                                intermediate_size=Validator.check_positive_int,
                                compute_dtype=_valid_value_checks([mstype.float32, mstype.float16],
                                                                  "FeedForward"),
                                param_init_type=_valid_value_checks([mstype.float32, mstype.float16],
                                                                    "FeedForward"))
    def __init__(self, dim,
                 intermediate_size=0,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32,
                 is_dynamic=False):
        super().__init__()

        hidden_dim = intermediate_size
        self.dtype = compute_dtype
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.mul = P.Mul()
        self.cast = P.Cast()
        self.silu = LlamaSiLU()

        self.w1 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)

        self.w2 = Linear(in_channels=hidden_dim,
                         out_channels=dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)

        self.w3 = Linear(in_channels=dim,
                         out_channels=hidden_dim,
                         has_bias=False,
                         compute_dtype=compute_dtype,
                         param_init_type=param_init_type,
                         skip_redistribution=is_dynamic)

    def construct(self, x):
        """Forward process of the FeedForward"""
        _check_input_dtype(F.dtype(x), "x", [mstype.float32, mstype.float16], self.cls_name)
        x = self.cast(x, self.dtype)
        # [bs, seq, hidden_dim] or [bs * seq, hidden_dim]
        gate = self.w1(x)  # dp,1 -> dp, mp
        hidden = self.w3(x)  # dp,1 -> dp, mp
        hidden = self.mul(gate, self.silu(hidden).astype(self.dtype))  # dp,mp -> dp, mp
        output = self.w2(hidden)  # dp,mp -> dp, 1
        return output

    def shard(self, parallel_config):
        """sharding for feedforward"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if self.hidden_dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'hidden_dim' must be a multiple of the"
                             "num of model parallel, but got the hidden_dim is {} and the num of model "
                             "parallel is {}.".format(self.hidden_dim, mp))
        if self.dim % mp != 0:
            raise ValueError("For 'FeedForward', the class variable 'dim' must be a multiple of the num of "
                             "model parallel, but got the dim is {} and the num of model parallel is {}."
                             .format(self.dim, mp))
        self.w1.shard(((dp, 1), (mp, 1)), strategy_activation=((dp, mp),))
        self.w2.shard(((dp, mp), (1, mp)))
        self.w3.shard(((dp, 1), (mp, 1)))
        self.mul.shard(((dp, mp), (dp, mp)))
        self.silu.shard(((dp, 1, mp),))


class CausalMaskForQwen(nn.Cell):
    r""" Get the Lower triangular matrix from the input_ids.
            [[[1. 0. 0. 0. 0]
              [1. 1. 0. 0. 0]
              [1. 1. 1. 0. 0]
              [1. 1. 1. 1. 0]
              [1. 1. 1. 1. 0]]]"""

    def __init__(self, seq_length, compute_type=mstype.float16,
                 is_dynamic=False, pad_token_id=0, use_flash_attention=False):
        super().__init__()
        self.dtype = compute_type
        self.is_dynamic = is_dynamic
        self.pad_token_id = pad_token_id
        self.use_flash_attention = use_flash_attention
        self.multiply_data = Tensor([-10000.0], dtype=compute_type)
        self.one = Tensor([1.0], dtype=compute_type)
        self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length))), mstype.float32)

        self.shape = P.Shape()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.not_equal = P.NotEqual()
        self.less_equal = P.LessEqual()
        self.expand_dim = P.ExpandDims()
        self.slice = P.StridedSlice()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.mul_post = P.Mul()
        self.expand_dim_post = P.ExpandDims()

    def construct(self, tokens):
        """Forward process of the CausalMask"""
        bs = self.shape(tokens)[0]
        seq_len = self.shape(tokens)[1]
        input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), self.dtype)
        shape_right = (bs, 1, seq_len)
        # Mask the padded inputs
        mask_right = self.reshape(input_mask, shape_right)
        if not self.is_dynamic:
            lower_traiangle = self.expand_dim(self.lower_triangle_mask, 0)
        else:
            lower_triangle_mask = self.slice(self.lower_triangle_mask, (0, 0), (seq_len, seq_len), (1, 1))
            lower_traiangle = self.expand_dim(lower_triangle_mask, 0)
        # the returned shape is [bs, seq_length, seq_length]
        attention_mask = self.mul(mask_right, lower_traiangle)
        return attention_mask

    def increment(self, seq_range, batch_valid_length, zactivate_len=None):
        if zactivate_len is not None:
            seq_range = self.slice(seq_range, (0, 0, 0), (1, 1, self.shape(zactivate_len)[0]), (1, 1, 1))
        mask = self.less_equal(self.reshape(seq_range, (1, 1, -1)), self.reshape(batch_valid_length, (-1, 1, 1)))
        return mask

    def increment_slice(self, seq_range, seq_length, batch_valid_length, zactivate_len=None):
        if zactivate_len is not None:
            seq_range_mask = self.slice(seq_range, (0, 0, 0), (1, 1, self.shape(zactivate_len)[0]), (1, 1, 1))
        else:
            seq_range_mask = self.slice(seq_range, (0, 0, 0), (1, 1, seq_length), (1, 1, 1))
        mask = self.less_equal(self.reshape(seq_range_mask, (1, 1, -1)), self.reshape(batch_valid_length, (-1, 1, 1)))
        return mask

    def post_process(self, mask):
        mask = self.sub(self.one, self.cast(mask, self.dtype))
        if not self.use_flash_attention:
            mask = self.expand_dim_post(mask, 1)
            mask = self.mul_post(mask, self.multiply_data)
        return mask

    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        self.not_equal.shard(((dp, 1), ()))
        self.expand_dim.shard(((1, 1),))
        self.mul.shard(((dp, 1, 1), (1, 1, 1)))
        self.less_equal.shard(((1, 1, 1), (1, 1, 1)))
        self.sub.shard(((1,), (dp, 1, 1)))
        self.mul_post.shard(((dp, 1, 1, 1), (1,)))
        self.expand_dim_post.shard(((dp, 1, 1),))
