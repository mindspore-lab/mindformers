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
"""iFlytekSpark model layers' APIs."""
import copy
import math
from typing import Callable, Optional
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Parameter
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.common.initializer import Initializer
from mindspore.common.initializer import _assignment
from mindspore.common.initializer import _calculate_fan_in_and_fan_out
from mindspore.common.initializer import _init_random_normal
from mindspore.common.initializer import _register
from mindspore.common.initializer import initializer
from mindspore.ops import functional as F

from mindformers.modules.layers import LayerNorm as _LayerNorm
from mindformers.modules.layers import Linear as _Linear
from mindformers.modules import KVCacheMgr
from mindformers.version_control import choose_flash_attention_dtype, is_910a

from iflytekspark_config import IFlytekSparkConfig


@_register('xavier_normal')
class XavierNormal(Initializer):
    r"""
    Generates an array with values sampled from Xavier normal distribution
    :math:`\mathcal{N}(0, \text{sigma}^2)` in order to initialize a tensor, where

    .. math::
        sigma = gain * \sqrt{\frac{2}{n_{in} + n_{out}}}

    where :math:`gain` is an optional scaling factor, :math:`n_{in}` is the number of input units in the weight tensor,
    :math:`n_{out}` is the number of output units in the weight tensor.

    Args:
        gain (float): An optional scaling factor. Default: 1.

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, XavierNormal
        >>> tensor1 = initializer(XavierNormal(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('xavier_normal', [1, 2, 3], mindspore.float32)
    """

    def __init__(self, gain=1):
        super(XavierNormal, set_parallel_configure_for_layer).__init__(gain=gain)
        self.gain = gain

    # pylint: disable=W0221
    def _initialize(self, arr):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(arr.shape)

        std = self.gain * math.sqrt(2.0 / float(fan_in + fan_out))
        data = _init_random_normal(0, std, arr.shape)

        _assignment(arr, data)


def set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
    r"""
        Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.


        Args:
            network(Cell) - Represents the transformer block
            layer_id(int) - Means the layer index for the current module, counts from zero.
            offset(int) - Means the layer_index needs a offset, if there are other modules in the net.
            layers(int) - The total layers used for the model.

    """
    num_layer = layers
    num_stage = parallel_config.pipeline_stage
    if num_stage == 10:
        mean_floor = num_layer // num_stage
        stage_layers_list = np.ones(num_stage) * mean_floor
        remaining_list = np.zeros(num_stage)
        remaining_list[0] = -1
        remaining_list[1] = -1
        remaining_list[5] = 1
        remaining_list[6] = 1
        remaining_list[7] = 1
        remaining_list[8] = 1
        stage_layers_list = remaining_list + stage_layers_list
        assert np.sum(stage_layers_list) == num_layer
        layer_list = [np.sum(stage_layers_list[:i + 1]) for i in range(len(stage_layers_list))]
        layer_list = np.array(layer_list)
        pp_id = int(np.sum(layer_list < layer_id + 1))
        network.pipeline_stage = pp_id
    else:
        # Used for the pipeline's stages setting
        # As the final layer is not included here, so we need to manually add here.
        # original:  if set two stages, layers on two stages will be [15, 16+1]
        # with 1 added, the layers on two stages will be [16, 15 +1]
        pp_dis = max(int((layers + 1) / parallel_config.pipeline_stage), 1)
        # the pipeline stage must be in [0, parallel_config.pipeline_stage - 1]
        pp_id = min((layer_id + offset) // pp_dis, parallel_config.pipeline_stage - 1)
        network.pipeline_stage = pp_id
    # print(f"layer {layer_id} pipeline stage id is {pp_id}", flush=True)
    # Used for optimizer's fusion tag
    dis = max(int((layers + 1) / parallel_config.gradient_aggregation_group), 1)
    if parallel_config.pipeline_stage > 1:
        # we give the fusion in pipeline mode a fixed value, otherwise the performance may become worse.
        network.set_comm_fusion(2)
    else:
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
    # Used for enabling recomputation of the block
    if parallel_config.recompute.recompute:
        # if layer_id != layers -1 or layer_id != 0:
        network.recompute(recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)


class VocabEmbedding(nn.Cell):
    """
        The embedding lookup table from the 0-th dim of the parameter table. When the parallel_config.vocab_emb_dp is
        True and in the `AUTO_PARALLEL` mode, the embedding lookup will be trained by the data parallel way, as the
        parameters will be repeated on each device. If false, the embedding table will be sharded into n parts at
        the 0-th dimension of the embedding table, where the n is the model parallel way determined by
        `parallel_config.model_parallel` (EmbeddingOpParallelConfig).

        Note:
            When `AUTO_PARALLEL` or `SEMI_AUTO_PARALLEL` mode is enabled, this layer support only 2-d dimension inputs,
            as the shard is designed for 2d inputs.

        Args:
            vocab_size (int): Size of the dictionary of embeddings.
            embedding_size (int): The size of each embedding vector.
            parallel_config (EmbeddingOpParallelConfig): The parallel config of network. Default
                `default_embedding_parallel_config`, an instance of `EmbeddingOpParallelConfig` with default args.
            param_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
                Refer to class `initializer` for the values of string when a string
                is specified. Default: 'normal'.

        Inputs:
            - **input_ids** (Tensor) - The tokenized inputs with datatype int32 with shape (batch_size, seq_length)

        Outputs:
            Tuple, a tuple contains (`output`, `embedding_table`)

            - **output** (Tensor) - The embedding vector for the input with shape (batch_size,
              seq_length, embedding_size).
            - **embedding_table** (Tensor) - The embedding table with shape (vocab_size, embedding_size).

        Raises:
            ValueError: If the parallel_config.vocab_emb_dp is True, the vocab size is not a multiple of
                parallel_config.model_parallel
            ValueError: `vocab_size` is not a positive value.
            ValueError: `embedding_size` is not a positive value.
            TypeError: `parallel_config` is not a subclass of OpParallelConfig.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore.nn.transformer import VocabEmbedding
            >>> from mindspore import Tensor
            >>> from mindspore import dtype as mstype
            >>> model = VocabEmbedding(vocab_size=30, embedding_size=30)
            >>> tensor = Tensor(np.ones((20, 15)), mstype.int32)
            >>> output, table = model(tensor)
            >>> print(output.shape)
            (20, 15, 30)
            >>> print(table.shape)
            (30, 30)
    """

    def __init__(self, vocab_size, embedding_size, padding_idx, init_method=XavierNormal, dtype=ms.float16,
                 parallel_config=None):
        super(VocabEmbedding, self).__init__()
        dtype = ms.float32 if dtype == ms.bfloat16 else dtype
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.init_tensor = initializer(init_method, [self.vocab_size, self.embedding_size], dtype)

        self.padding_idx = padding_idx
        if padding_idx is not None:
            if isinstance(self.init_tensor, Tensor) and self.init_tensor.init is not None:
                self.init_tensor = self.init_tensor.init_data()
            self.init_tensor = self.init_tensor.asnumpy()
            self.init_tensor[self.padding_idx] = 0
            self.init_tensor = Tensor(self.init_tensor)

        self.embedding_table = Parameter(self.init_tensor, name='embedding_table', parallel_optimizer=False)
        if parallel_config.vocab_emb_dp:
            self.gather = P.Gather().shard(((1, 1), (parallel_config.data_parallel, 1)))
        else:
            if self.vocab_size % parallel_config.model_parallel != 0:
                raise ValueError(f"The vocab size of the embedding {self.vocab_size} must be a "
                                 f"multiple of parallel_config.model_parallel {parallel_config.model_parallel}.")
            self.gather = P.Gather().shard(((parallel_config.model_parallel, 1), (parallel_config.data_parallel, 1)))

    # pylint: disable=W0221
    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table


class RotaryPositionalTransform(nn.Cell):
    """ Rotary positional embedding"""
    def __init__(self, dim, parallel_config, seq_len=8192, base=1000000,
                 use_past=False, is_dynamic=False, compute_type=ms.float16):
        super(RotaryPositionalTransform, self).__init__()
        self.inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, np.float32) / dim))
        self.use_past = use_past
        self.dim = dim
        self.is_dynamic = is_dynamic

        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, self.inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = Tensor(np.cos(emb), compute_type)
        self.sin_cached = Tensor(np.sin(emb), compute_type)

        mask = np.eye(dim, k=dim // 2) - np.eye(dim, k=-dim // 2)  # head_dim, head_dim
        self.rotary_mask = Tensor(mask, dtype=compute_type)
        self.bmm = P.BatchMatMul().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), (1, 1)))
        self.add = P.Add().shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                                  (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.mul = P.Mul().shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), (1, 1)))
        self.shape = P.Shape()
        if use_past:
            self.sub = P.Sub().shard(((1, 1), ()))
            self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
            self.expand_dims1 = P.ExpandDims().shard(((1, 1),))
            self.slice = P.StridedSlice().shard(((1, 1),))
            self.gather = P.Gather().shard(((1, 1), (1, 1)))
            self.mul = P.Mul().shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                                      (1, 1, 1, 1)))

    # pylint: disable=W0221
    def construct(self, q, k, batch_valid_length=None):
        """
        q: B x h_num X T_q  X  C
        k: B X h_num x T    X  C
        while training, T_q=T, while inference, T_q=1
        """
        _, _, seq_length, _ = self.shape(k)

        if self.use_past and self.is_first_iteration and self.is_dynamic:
            cos = self.slice(self.cos_cached, (0, 0), (seq_length, self.dim), (1, 1))
            sin = self.slice(self.sin_cached, (0, 0), (seq_length, self.dim), (1, 1))
        else:
            cos = self.cos_cached
            sin = self.sin_cached

        if self.use_past and self.is_first_iteration:
            cos = self.expand_dims(self.expand_dims1(cos, 0), 0)
            sin = self.expand_dims(self.expand_dims1(sin, 0), 0)

        if self.use_past and not self.is_first_iteration and batch_valid_length is not None:
            batch_valid_length = self.sub(batch_valid_length.reshape((-1, 1)), 1)
            cos = self.expand_dims(self.gather(cos, batch_valid_length, 0), 1)
            sin = self.expand_dims(self.gather(sin, batch_valid_length, 0), 1)

        if q.shape[1] > k.shape[1]:
            raise ValueError(f"q shape {q.shape[1]} bigger than k {k.shape[1]}")
        if q.shape[1] < k.shape[1]:
            cos_q, sin_q = cos[-q.shape[1]:], sin[-q.shape[1]:]
        else:
            cos_q, sin_q = cos, sin

        q = self.add(self.mul(q, cos_q), self.mul(self.rotate_half(q), sin_q))
        k = self.add(self.mul(k, cos), self.mul(self.rotate_half(k), sin))
        return q, k

    # rotary pos emb helpers:
    def rotate_half(self, x):
        # x: b, head_num, T, H_D
        rotary = self.bmm(x, self.rotary_mask)  # to avoid split ops
        return rotary

class IFlytekSparkKVCacheMgr(KVCacheMgr):
    """ iFlytekSpark kv cache mgr for ms lite"""
    def __init__(self,
                 n_head,
                 head_dim,
                 ori_seq_length=8192,
                 max_batch_size=8,
                 max_seq_length=4096,
                 compute_dtype=mstype.float16,
                 is_dynamic=False,
                 use_kvcache_op=True,
                 is_flexible_shape=False,
                 parallel_config=None):
        super(IFlytekSparkKVCacheMgr, self).__init__(n_head, head_dim, max_batch_size, max_seq_length,
                                                     compute_dtype, is_dynamic, use_kvcache_op, is_flexible_shape)
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.ori_seq_length = ori_seq_length
        self.prompt_kvcache.shard(((dp, mp, 1, 1), (dp, mp, 1, 1), (dp,), (1,), (1,), (1,), (1,)))
        self.decoder_kvcache.shard(((dp, mp, 1, 1), (dp, mp, 1, 1), (dp,), (1,), (1,), (1,), (1,)))
        self.slice.shard(((dp, mp, 1, 1),))
        self.pad = P.PadV3().shard(((dp, mp, 1, 1), (1,), ()))
        self.pad1 = P.PadV3().shard(((1,), (1,), ()))
        self.sub = P.Sub().shard(((1,), ()))
        self.mul = P.Mul().shard(((1,), (1,)))
        self.add = P.Add().shard(((1,), (1, 1)))
        self.gather = P.Gather(1).shard(((1, 1, 1, 1), (1, 1)))
        self.concat = P.Concat(axis=0)
        self.seq_idx = Tensor(np.arange(self.max_seq_length), mstype.int32)
        self.ori_seq_length_tensor = Tensor([ori_seq_length], mstype.int32)
        self.cache_length_tensor = Tensor([self.max_seq_length], mstype.int32)
        self.cache_batch_tensor = Tensor([self.max_batch_size], mstype.int32)
        self.tensor_1 = Tensor([0,], mstype.int32)
        self.tensor_2 = Tensor([0, 0], mstype.int32)
        self.tensor_3 = Tensor([0, 0, 0], mstype.int32)
        self.tensor_6 = Tensor([0, 0, 0, 0, 0, 0], mstype.int32)
        self.tensor_zero = Tensor(0, mstype.int64)


    def padding(self, k, v, bsz, seq_len, batch_valid_length):
        """padding."""
        if self.is_first_iteration:
            max_len = self.ori_seq_length_tensor
        else:
            max_len = self.cache_length_tensor
        kv_pad_length = self.sub(max_len, seq_len).reshape((1,)).astype(mstype.int32)
        batch_pad_length = self.sub(self.cache_batch_tensor, bsz).reshape((1,)).astype(mstype.int32)
        # calculate padding parameter: (0, batch),(0, 0), (0, pad_length), (0, 0)
        if self.is_first_iteration:
            kv_paddings = self.concat((self.tensor_1, batch_pad_length,
                                       self.tensor_3, kv_pad_length,
                                       self.tensor_2))
        else:
            kv_paddings = self.concat((self.tensor_1, batch_pad_length,
                                       self.tensor_6))
        batch_paddings = self.concat((self.tensor_1, batch_pad_length))
        k_pad = self.pad(k, kv_paddings, self.pad_zero)
        v_pad = self.pad(v, kv_paddings, self.pad_zero)
        batch_valid_length_pad = self.pad1(batch_valid_length.view(-1,),
                                           batch_paddings, self.tensor_zero)
        return k_pad, v_pad, batch_valid_length_pad

    def _slice_input_to_cache_size(self, x, batch_valid_length):
        if self.ori_seq_length > self.max_seq_length:
            over_buffer = batch_valid_length > self.max_seq_length
            shift_value = self.sub(batch_valid_length, self.max_seq_length)
            shift_value = self.mul(shift_value, over_buffer.astype(ms.int32))
            index = self.add(self.seq_idx, shift_value.reshape((-1, 1)))
            x = self.gather(x, index, 2)
        return x

    def auto_caching(self, key_update, value_update, batch_valid_length, seq_length_tensor_pad, batch_index_pad=None):
        """use kvcache op to cache key, value"""
        # key_update shape: [real_bs, n_head, max_seqlen, head_dim]
        if self.is_first_iteration:
            save_start_point = batch_valid_length * self.tensor_zero
            key_update = self._slice_input_to_cache_size(key_update, batch_valid_length)
            value_update = self._slice_input_to_cache_size(value_update, batch_valid_length)
            self.prompt_kvcache(self.key_past, key_update, save_start_point, batch_index_pad,
                                self.seqlen_axis_tensor_pad, seq_length_tensor_pad, seq_length_tensor_pad)
            self.prompt_kvcache(self.value_past, value_update, save_start_point, batch_index_pad,
                                self.seqlen_axis_tensor_pad, seq_length_tensor_pad, seq_length_tensor_pad)
            return self.tensor_zero, self.tensor_zero # to avoid return None

        key_cache = self.key_past
        value_cache = self.value_past
        current_index = self.cast(self.sub(batch_valid_length, 1), mstype.int64)
        self.decoder_kvcache(self.key_past, key_update, current_index, batch_index_pad,
                             self.seqlen_axis_tensor_pad, seq_length_tensor_pad, seq_length_tensor_pad)
        self.decoder_kvcache(self.value_past, value_update, current_index, batch_index_pad,
                             self.seqlen_axis_tensor_pad, seq_length_tensor_pad, seq_length_tensor_pad)
        key_cache = P.depend(key_cache, key_update)
        value_cache = P.depend(value_cache, value_update)
        return key_cache, value_cache

    def construct(self, key, value, batch_valid_length,
                  zactivate_len=None, batch_index_pad=None, seq_length_tensor_pad=None):
        """The forward compute of KVCacheMgr."""
        batch_size, _, seq_length, _ = self.shape(key)
        if self.is_dynamic:
            key, value, batch_valid_length = self.padding(key, value, batch_size,
                                                          seq_length, batch_valid_length)
        key, value = self.auto_caching(key, value, batch_valid_length,
                                       seq_length_tensor_pad, batch_index_pad)
        if not self.is_first_iteration:
            zactivate_len = None
            key, value = self.trimming(key, value, zactivate_len, batch_size=batch_size)

        return key, value


class IFlytekSparkAttention(nn.Cell):
    """ iFlytekSpark core attention layer"""
    def __init__(
            self,
            cfg,
            head_dim: int,
            num_head: int,
            rank_id: int,
            layer_scaling: Optional[int] = -1,
            self_attention: Optional[bool] = False,
            encoder_decoder_attention: Optional[bool] = False,
            use_rope=False
    ) -> None:
        super(IFlytekSparkAttention, self).__init__()
        self.head_dim = head_dim
        self.num_head = num_head
        self.layer_scaling = max(1, layer_scaling)
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.curr_rank = rank_id
        self.softmax_compute_type = cfg.softmax_compute_type
        self.use_rope = use_rope
        self.use_past = cfg.use_past
        self.compute_type = cfg.compute_type

        self.rotary_embedding = None
        if self.use_rope:
            self.rotary_embedding = RotaryPositionalTransform(self.head_dim,
                                                              cfg.parallel_config,
                                                              cfg.seq_length,
                                                              use_past=cfg.use_past,
                                                              is_dynamic=cfg.is_dynamic,
                                                              compute_type=self.compute_type)

        self.bmm_q_k = ms.ops.BatchMatMul(transpose_b=True)
        self.bmm_q_k.shard(((cfg.parallel_config.data_parallel, cfg.parallel_config.model_parallel, 1, 1),
                            (cfg.parallel_config.data_parallel, cfg.parallel_config.model_parallel, 1, 1)))
        self.bmm = ms.ops.BatchMatMul()
        self.bmm.shard(((cfg.parallel_config.data_parallel, cfg.parallel_config.model_parallel, 1, 1),
                        (cfg.parallel_config.data_parallel, cfg.parallel_config.model_parallel, 1, 1)))
        self.cast1 = P.Cast().shard(((cfg.parallel_config.data_parallel, 1, 1),))
        self.cast2 = P.Cast().shard(((1,),))
        self.transpose = P.Transpose().shard(
            ((cfg.parallel_config.data_parallel, 1, cfg.parallel_config.model_parallel, 1),))
        self.merger_head_transpose = P.Transpose().shard(
            ((cfg.parallel_config.data_parallel, cfg.parallel_config.model_parallel, 1, 1),))
        self.multiply_data = Tensor([-10000.0,], dtype=self.softmax_compute_type)

        self.expand_dims = P.ExpandDims().shard(((cfg.parallel_config.data_parallel, 1, 1),))
        self.add = P.Add().shard(((cfg.parallel_config.data_parallel, 1, 1, 1),
                                  (cfg.parallel_config.data_parallel, cfg.parallel_config.model_parallel, 1, 1)))
        self.mul = P.Mul().shard(((cfg.parallel_config.data_parallel, 1, 1, 1), (1,)))

        self.softmax = P.Softmax().shard(
            ((cfg.parallel_config.data_parallel, cfg.parallel_config.model_parallel, 1, 1),))

        self.attention_dropout = nn.Dropout(p=cfg.dropout_rate)
        self.attention_dropout.dropout.shard(
            ((cfg.parallel_config.data_parallel, cfg.parallel_config.model_parallel, 1, 1),))
        self.shape = P.Shape()

        # import flash attention
        dp = cfg.parallel_config.data_parallel
        mp = cfg.parallel_config.model_parallel
        self.scale_factor = Tensor(math.sqrt(math.sqrt(self.head_dim)), dtype=self.compute_type)
        self.real_div = P.RealDiv().shard(((dp, mp, 1, 1), ()))
        self.is_910a = is_910a()
        if self.is_910a:
            from acctransformer.flash_attention.nn.layer.flash_attention import FlashAttention
            pre_blocks = (cfg.sparse_local_size // 128) - 1
            self.attention_mask_dtype = choose_flash_attention_dtype()
            self.flash_attention = FlashAttention(head_dim, num_head,
                                                  cfg.dropout_rate, pre_blocks, 0,
                                                  dp=dp, mp=mp, high_precision=True)
        else:
            from mindspore.nn.layer.flash_attention import FlashAttention
            from mindspore.ops.operations.nn_ops import PromptFlashAttention
            pre_tokens = cfg.sparse_local_size - 1
            self.attention_mask_dtype = choose_flash_attention_dtype()
            self.flash_attention = FlashAttention(head_dim, num_head, cfg.dropout_rate, pre_tokens, 0,
                                                  dp=dp, mp=mp, high_precision=True)
            self.prompt_flash_attention = PromptFlashAttention(num_heads=self.num_head,
                                                               scale_value=1.0,
                                                               pre_tokens=pre_tokens,
                                                               next_tokens=0,
                                                               input_layout="BNSD",
                                                               num_key_value_heads=0)
            q_shard_stgy = k_shard_stgy = v_shard_stgy = (dp, mp, 1, 1)
            attn_mask_shard_stgy = (dp, 1, 1, 1)
            in_stgy = (q_shard_stgy, k_shard_stgy, v_shard_stgy, attn_mask_shard_stgy)
            self.prompt_flash_attention.shard(in_stgy)
        if not cfg.flash_attention_recompute:
            self.flash_attention.flash_attention.recompute(False)
            if hasattr(self.flash_attention, 'drop_gen_mask'):
                self.flash_attention.drop_gen_mask.recompute(False)

        self.is_lite_infer = cfg.is_lite_infer
        self.is_dynamic = cfg.is_dynamic
        self.seq_length = cfg.seq_length
        if self.use_past:
            # operators used for state reuse
            self.max_cache_length = self.seq_length \
                if self.seq_length <= cfg.sparse_local_size else cfg.sparse_local_size
            self.max_cache_batch_size = cfg.batch_size
            self.one = Tensor((1.0,), dtype=self.compute_type)
            seq_range_prefill = np.arange(self.seq_length)
            seq_range_decode = np.arange(self.max_cache_length)
            self.range_decode = Tensor(np.tile(seq_range_decode.reshape(1, 1, -1),
                                               (self.max_cache_batch_size, 1, 1)), mstype.int32)
            self.range_prefill = Tensor(np.tile(seq_range_prefill.reshape(1, 1, -1),
                                                (self.max_cache_batch_size, 1, 1)), mstype.int32)
            self.seq_idx = Tensor(seq_range_decode, mstype.int32)
            self.expand_dims1 = P.ExpandDims().shard(((1, 1, 1),))
            self.add1 = P.Add().shard(((1,), (1,)))
            self.add2 = P.Add().shard(((1,), (1, 1)))
            self.add3 = P.Add().shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
            self.sub = P.Sub().shard(((1,), ()))
            self.sub1 = P.Sub().shard(((1,), (1,)))
            self.sub2 = P.Sub().shard(((1,), (1, 1)))
            self.sub3 = P.Sub().shard(((1, 1), ()))
            self.sub4 = P.Sub().shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
            self.mul1 = P.Mul().shard(((dp, mp, 1, 1), (1, 1, 1, 1)))
            self.mul2 = P.Mul().shard(((1,), ()))
            self.mul3 = P.Mul().shard(((1,), (1,)))
            self.gather = P.Gather(1).shard(((dp, 1, 1), (1, 1)))
            self.gather1 = P.Gather(1).shard(((dp, mp, 1, 1), (dp, 1)))
            self.gather2 = P.Gather(1).shard(((dp, mp, 1, 1), (dp, 1)))
            self.mod = P.FloorMod().shard(((1,), ()))
            self.mod1 = P.FloorMod().shard(((1, 1), ()))
            self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))

            if self.is_lite_infer:
                print("----- [Ms Lite Infer Now] -----")
                self.kvcache_mgr = IFlytekSparkKVCacheMgr(self.num_head, self.head_dim,
                                                          self.seq_length,
                                                          self.max_cache_batch_size,
                                                          self.max_cache_length,
                                                          compute_dtype=self.compute_type,
                                                          is_dynamic=self.is_dynamic,
                                                          use_kvcache_op=True,
                                                          is_flexible_shape=False,
                                                          parallel_config=cfg.parallel_config)

    # pylint: disable=W0221
    def construct(
            self,
            q, k, v,
            attention_mask,
            key_past=None,
            value_past=None,
            batch_valid_length=None,
            zactivate_len=None,
            batch_index_pad=None,
            seq_length_tensor_pad=None
    ):
        """Forward process of MultiHead Attention"""
        bsz, tgt_len, _ = self.shape(q)
        bsz, src_len, _ = self.shape(k)

        # [batch, len, num_head * head_dim] -> [batch , num_heads, seq_length, head_dim]
        q = self.transpose(
            F.reshape(q, (bsz, self._get_seq_length_under_incremental(tgt_len), self.num_head, self.head_dim)),
            (0, 2, 1, 3))

        # [batch, len, num_head * head_dim] -> [batch , num_heads, seq_length, head_dim]
        k = self.transpose(
            F.reshape(k, (bsz, self._get_seq_length_under_incremental(src_len), self.num_head, self.head_dim)),
            (0, 2, 1, 3))

        # [batch, len, num_head * head_dim] -> [batch , num_heads, seq_length, head_dim]
        v = self.transpose(
            F.reshape(v, (bsz, self._get_seq_length_under_incremental(src_len), self.num_head, self.head_dim)),
            (0, 2, 1, 3))

        if not self.training:
            batch_valid_length = F.reshape(batch_valid_length, (-1,))

        # rotary position embedding
        if self.use_rope:
            q, k = self.rotary_embedding(q, k, batch_valid_length=batch_valid_length)

        key_present = k
        value_present = v

        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # prefill
                key_present, value_present = self._prefill_kv_cache(k, v, batch_valid_length, zactivate_len,
                                                                    batch_index_pad, seq_length_tensor_pad)
            else:
                # decode
                key_present, value_present = self._decode_kv_cache(k, v, key_past, value_past,
                                                                   batch_valid_length, zactivate_len,
                                                                   batch_index_pad, seq_length_tensor_pad)
                k, v = key_present, value_present
        layer_present = (key_present, value_present)

        output = None
        if self.training:
            attention_mask = self.cast1(attention_mask, self.attention_mask_dtype)
            output = self.flash_attention(q, k, v, attention_mask)
        else:
            output = self._infer_attention(q, k, v, attention_mask, batch_valid_length)

        output = self._merge_heads(output)
        return output, layer_present

    def _infer_attention(self, q, k, v, attention_mask, batch_valid_length):
        """apply different attention in infer process."""
        factor = self.cast2(self.scale_factor, q.dtype)
        if self.is_910a:
            if self.use_past and self.is_first_iteration and not self.is_lite_infer:
                output = self.flash_attention(q, k, v, attention_mask)
            else:
                q = self.real_div(q, factor)
                k = self.real_div(k, factor)
                output = self.dense_attention(q, k, v, attention_mask, batch_valid_length)
        else:
            q = self.real_div(q, factor)
            k = self.real_div(k, factor)
            if self.use_past and not self.is_first_iteration:
                output = self.dense_attention(q, k, v, attention_mask, batch_valid_length)
            else:
                attention_mask = self.cast1(attention_mask, mstype.float16)
                attention_mask = self.expand_dims(attention_mask, 1)
                output = self.prompt_flash_attention(q, k, v, attention_mask,
                                                     None, None, None, None, None, None, None, None)[0]
        return output

    def _get_seq_length_under_incremental(self, length):
        r"""Return the length of the tensor.
            For the incremental prediction, the seq length for the input is 1.
        """
        if self.use_past and not self.is_first_iteration:
            return 1
        return length

    def _merge_heads(self, x):
        """ convert a 4d input to a 3d output"""
        output = self.merger_head_transpose(x, (0, 2, 1, 3))
        bsz, seq_len, _, _ = self.shape(output)
        new_shape = (bsz, seq_len, self.num_head * self.head_dim)
        output = F.reshape(output, new_shape)
        return output

    def _slice_input_to_cache_size(self, x, batch_valid_length):
        over_buffer = batch_valid_length > self.max_cache_length
        shift_value = self.sub(batch_valid_length, self.max_cache_length)
        shift_value = self.mul3(shift_value, over_buffer.astype(ms.int32))
        index = self.add2(self.seq_idx, shift_value.reshape((-1, 1)))
        x = self.gather2(x, index, 2)
        return x

    def _prefill_kv_cache(self, k, v, batch_valid_length, zactivate_len=None,
                          batch_index_pad=None, seq_length_tensor_pad=None):
        r"""Return prefill kv cache."""
        if self.is_lite_infer:
            key_present, value_present = self.kvcache_mgr(k, v, batch_valid_length, zactivate_len,
                                                          batch_index_pad, seq_length_tensor_pad)
        else:
            valid_length_vector = (self.less(
                self.range_prefill, F.reshape(batch_valid_length, (-1, 1, 1)))).astype(self.compute_type)
            # Cover the key and value numbers corresponding to the padding position
            key_present = self.mul1(k, self.expand_dims1(valid_length_vector, 3))
            value_present = self.mul1(v, self.expand_dims1(valid_length_vector, 3))
            if self.max_cache_length != self.seq_length:
                key_present = self._slice_input_to_cache_size(key_present, batch_valid_length)
                value_present = self._slice_input_to_cache_size(value_present, batch_valid_length)
                key_present, value_present = self._roll_kv_cache(key_present,
                                                                 value_present,
                                                                 batch_valid_length)

        return key_present, value_present

    def _decode_kv_cache(self, k, v, key_past, value_past, batch_valid_length,
                         zactivate_len=None, batch_index_pad=None, seq_length_tensor_pad=None):
        r"""Return decode kv cache """
        if self.is_lite_infer:
            key_present, value_present = self.kvcache_mgr(k, v, batch_valid_length, zactivate_len,
                                                          batch_index_pad, seq_length_tensor_pad)
        else:
            current_index = self.sub(batch_valid_length, 1)
            if self.max_cache_length != self.seq_length:
                current_index = self.mod(current_index, self.max_cache_length)

            current_index = F.reshape(current_index, (-1, 1, 1))
            current_mask = self.equal(self.range_decode, current_index).astype(self.compute_type)
            # Pad the key and value to seq_length with only the position index not zero
            current_key = self.mul1(k, self.expand_dims1(current_mask, 3))
            current_value = self.mul1(v, self.expand_dims1(current_mask, 3))
            if self.max_cache_length != self.seq_length:
                key_past = self.sub4(key_past, self.mul1(key_past, self.expand_dims1(current_mask, 3)))
                value_past = self.sub4(value_past, self.mul1(value_past, self.expand_dims1(current_mask, 3)))
            # Concat the previous saved state and current state
            key_present = self.add3(current_key, key_past)
            value_present = self.add3(current_value, value_past)
        if self.max_cache_length != self.seq_length:
            key_present, value_present = self._roll_kv_cache(key_present,
                                                             value_present,
                                                             batch_valid_length)

        return key_present, value_present

    def _roll_kv_cache(self, key_present, value_present, batch_valid_length):
        """roll key and value cache depend on batch_valid_length"""
        over_buffer = batch_valid_length > self.max_cache_length
        shift_value = self.mul2(self.mod(batch_valid_length, self.max_cache_length), 2)
        shift_value = self.sub1(batch_valid_length, shift_value)
        shift_value = self.mul3(shift_value, over_buffer.astype(mstype.int32))
        if self.is_first_iteration:
            index = self.add2(self.seq_idx, F.reshape(shift_value, (-1, 1)))
        else:
            index = self.sub2(self.seq_idx, F.reshape(shift_value, (-1, 1)))
        index = self.mod1(index, self.max_cache_length)
        key_present = self.gather1(key_present, index, 2)
        value_present = self.gather1(value_present, index, 2)
        return key_present, value_present

    def _batch_valid_length_correction(self, batch_valid_length):
        """correct batch_valid_length"""
        if self.max_cache_length != self.seq_length:
            over_buffer = self.mul2(batch_valid_length > self.max_cache_length, self.max_cache_length)
            below_buffer = self.mul3(batch_valid_length <= self.max_cache_length, batch_valid_length)
            batch_valid_length = self.add1(below_buffer, over_buffer)
        return batch_valid_length

    def dense_attention(self, q, k, v, attention_mask, batch_valid_length=None):
        """dense attention"""
        # q : [batch, num_heads, t, head_dim]
        # k : [batch, num_heads, t, head_dim]
        # attn_weights : [batch, num_heads, t, t]
        attn_weights = self.bmm_q_k(q, k)

        if attention_mask is not None:
            # b, 1, t, t
            if self.use_past and not self.is_first_iteration:
                batch_valid_length = self._batch_valid_length_correction(batch_valid_length)
                batch_valid_length = self.sub3(F.reshape(batch_valid_length, (-1, 1)), 1)
                attention_mask = self.gather(attention_mask, batch_valid_length, 1)
            attention_mask = self.expand_dims(attention_mask, 1)
            adder = self.mul(attention_mask, self.multiply_data)
            attn_weights = self.add(adder, attn_weights)

        attention_probs = self.softmax(attn_weights)
        attention_probs = self.attention_dropout(attention_probs)

        # Context layer
        assert v is not None
        output = self.bmm(attention_probs.astype(self.compute_type), v.astype(self.compute_type))

        return output


class IFlytekSparkEmbedding(nn.Cell):
    """iFlytekSpark Embedding module."""
    def __init__(
            self,
            embed_dim,
            vocab_size,
            max_sequence_length,
            embedding_dropout_prob,
            dtype,
            init_method,
            with_position=False,
            parallel_config=None
    ):
        super(IFlytekSparkEmbedding, self).__init__()

        self.embed_dim = embed_dim
        self.init_method = init_method
        self.with_position = with_position
        self.embedding_type = dtype

        # Word embeddings (parallel).
        self.word_embeddings = VocabEmbedding(
            vocab_size, self.embed_dim, None, dtype=self.embedding_type,
            init_method=self.init_method, parallel_config=parallel_config.embedding_dp_mp_config,
        )

        # Position embedding
        if self.with_position:
            copied_parallel_config = copy.deepcopy(parallel_config)
            copied_parallel_config.vocab_emb_dp = True
            self.position_embeddings = VocabEmbedding(vocab_size=vocab_size,
                                                      embedding_size=self.embed_dim,
                                                      padding_idx=None,
                                                      dtype=self.embedding_type,
                                                      init_method=initializer(init_method,
                                                                              [max_sequence_length, embed_dim],
                                                                              dtype=self.embedding_type),
                                                      parallel_config=copied_parallel_config.embedding_dp_mp_config)

            self.add = P.Add().shard(((parallel_config.data_parallel, 1, 1), (parallel_config.data_parallel, 1, 1)))
        # Embeddings dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout_prob)
        self.embedding_dropout.dropout.shard(((parallel_config.data_parallel, 1, 1),))
        self.cast = P.Cast().shard(((parallel_config.data_parallel, 1, 1),))

    # pylint: disable=W0221
    def construct(self, tokens, position_ids):
        """Forward process of Embedding"""
        # Embeddings.
        words_embeddings, words_embedding_table = self.word_embeddings(tokens)

        if self.with_position:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = self.add(words_embeddings, position_embeddings)
        else:
            embeddings = words_embeddings

        # Dropout.
        embeddings = self.cast(embeddings, self.embedding_type)
        embeddings = self.embedding_dropout(embeddings)
        return embeddings, words_embedding_table


class IFlytekSparkMultiheadAttention(nn.Cell):
    """ iFlytekSpark attention """
    def __init__(
            self,
            config: IFlytekSparkConfig,
            layer_number: int,
            rank_id: int,
            self_attention: Optional[bool] = False,
            encoder_decoder_attention: Optional[bool] = False,
            skip_last_bias_add: bool = True,
    ):

        super(IFlytekSparkMultiheadAttention, self).__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.kdim = config.hidden_size
        self.vdim = config.hidden_size
        self.bias = True
        self.layer_number = layer_number
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        # cfg.common.fp16
        self.compute_type = config.compute_type
        self.recompute = config.parallel_config.recompute
        if config.seq_parallel:
            self.recompute = False

        self.head_dim = self.embed_dim // self.num_heads

        if self.self_attention:
            self.q_k_v_proj = IFlytekSparkLinear(self.embed_dim, self.embed_dim * 3,
                                                 bias=self.bias, compute_type=self.compute_type)
            self.q_k_v_proj.shard(strategy_matmul=((config.parallel_config.data_parallel, 1),
                                                   (config.parallel_config.model_parallel, 1)),
                                  strategy_bias=((config.parallel_config.data_parallel,
                                                  config.parallel_config.model_parallel),
                                                 (config.parallel_config.model_parallel,)))
            self.split3 = P.Split(-1, 3).shard(((config.parallel_config.data_parallel, 1, 1),))
        elif self.encoder_decoder_attention:
            self.q_proj = IFlytekSparkLinear(self.embed_dim, self.embed_dim,
                                             bias=self.bias, compute_type=self.compute_type)
            self.q_proj.shard(
                strategy_matmul=((config.parallel_config.data_parallel, 1), (config.parallel_config.model_parallel, 1)),
                strategy_bias=((config.parallel_config.data_parallel, config.parallel_config.model_parallel),
                               (config.parallel_config.model_parallel,)))
            if self.vdim == self.kdim:
                self.k_v_proj = IFlytekSparkLinear(self.kdim, self.embed_dim * 2,
                                                   bias=self.bias, compute_type=self.compute_type)
                self.k_v_proj.shard(
                    strategy_matmul=((config.parallel_config.data_parallel, 1),
                                     (config.parallel_config.model_parallel, 1)),
                    strategy_bias=((config.parallel_config.data_parallel, config.parallel_config.model_parallel),
                                   (config.parallel_config.model_parallel,)))
                self.split2 = P.Split(-1, 2).shard(((config.parallel_config.data_parallel, 1, 1),))
            else:
                self.k_proj = IFlytekSparkLinear(self.embed_dim,
                                                 self.embed_dim,
                                                 bias=self.bias,
                                                 compute_type=self.compute_type)
                self.k_proj.shard(
                    strategy_matmul=((config.parallel_config.data_parallel, 1),
                                     (config.parallel_config.model_parallel, 1)),
                    strategy_bias=((config.parallel_config.data_parallel, config.parallel_config.model_parallel),
                                   (config.parallel_config.model_parallel,)))

                self.v_proj = IFlytekSparkLinear(self.embed_dim,
                                                 self.embed_dim,
                                                 bias=self.bias,
                                                 compute_type=self.compute_type)
                self.v_proj.shard(
                    strategy_matmul=((config.parallel_config.data_parallel, 1),
                                     (config.parallel_config.model_parallel, 1)),
                    strategy_bias=((config.parallel_config.data_parallel, config.parallel_config.model_parallel),
                                   (config.parallel_config.model_parallel,)))
        else:
            self.q_proj = IFlytekSparkLinear(self.embed_dim, self.embed_dim,
                                             bias=self.bias, compute_type=self.compute_type)
            self.q_proj.shard(strategy_matmul=((config.parallel_config.data_parallel, 1),
                                               (config.parallel_config.model_parallel, 1)),
                              strategy_bias=((config.parallel_config.data_parallel,
                                              config.parallel_config.model_parallel),
                                             (config.parallel_config.model_parallel,)))

            self.k_proj = IFlytekSparkLinear(self.embed_dim, self.embed_dim,
                                             bias=self.bias, compute_type=self.compute_type)
            self.k_proj.shard(strategy_matmul=((config.parallel_config.data_parallel, 1),
                                               (config.parallel_config.model_parallel, 1)),
                              strategy_bias=((config.parallel_config.data_parallel,
                                              config.parallel_config.model_parallel),
                                             (config.parallel_config.model_parallel,)))

            self.v_proj = IFlytekSparkLinear(self.embed_dim, self.embed_dim,
                                             bias=self.bias, compute_type=self.compute_type)
            self.v_proj.shard(strategy_matmul=((config.parallel_config.data_parallel, 1),
                                               (config.parallel_config.model_parallel, 1)),
                              strategy_bias=((config.parallel_config.data_parallel,
                                              config.parallel_config.model_parallel),
                                             (config.parallel_config.model_parallel,)))

        self.core_attention = IFlytekSparkAttention(
            config,
            head_dim=self.head_dim,
            num_head=self.num_heads,
            rank_id=rank_id,
            layer_scaling=self.layer_number,
            self_attention=self.self_attention,
            encoder_decoder_attention=self.encoder_decoder_attention,
            use_rope=True
        )
        # input : dp,1,mp -> output dp, 1, 1
        self.out_proj = IFlytekSparkLinear(self.embed_dim, self.embed_dim, bias=self.bias, transpose_b=True,
                                           skip_bias_add=skip_last_bias_add, compute_type=self.compute_type)
        self.out_proj.shard(strategy_bias=((config.parallel_config.data_parallel, 1), (1,)),
                            strategy_matmul=((config.parallel_config.data_parallel,
                                              config.parallel_config.model_parallel),
                                             (1, config.parallel_config.model_parallel)))
        if config.seq_parallel:
            self.out_proj.matmul.shard(((config.parallel_config.data_parallel, config.parallel_config.model_parallel),
                                        (1, config.parallel_config.model_parallel)),
                                       ((config.parallel_config.data_parallel *
                                         config.parallel_config.model_parallel, 1),))
            self.out_proj.bias_add.shard(
                ((config.parallel_config.data_parallel * config.parallel_config.model_parallel, 1), (1,)))

    # pylint: disable=W0221
    def construct(
            self,
            q,
            k,
            v,
            attention_mask,
            key_past=None,
            value_past=None,
            batch_valid_length=None,
            zactivate_len=None,
            batch_index_pad=None,
            seq_length_tensor_pad=None
    ):
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attention_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attention_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
        """
        if self.self_attention:
            # [len, batch, num_head * head_dim] -> [len, batch, num_head * 3 * head_dim]
            q_k_v = self.q_k_v_proj(q)
            # [len, batch, num_head, 3 * head_dim] -> 3 * [len, batch, num_head * head_dim]
            q, k, v = self.split3(q_k_v)
        elif self.encoder_decoder_attention:
            q = self.q_proj(q)
            if k is None:
                assert v is None
                k = v = None
            else:
                if self.kdim == self.vdim:
                    k_v = self.k_v_proj(k)
                    k, v = self.split2(k_v)
                else:
                    # multi head attention: query, key, value are derived from the same inputs
                    k = self.k_proj(k)
                    v = self.v_proj(v)
        else:
            assert k is not None and v is not None
            # multi head attention: query, key, value are derived from the same inputs
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)

        # =======================================
        # Compute attention scores.
        # # =======================================
        context, layer_present = self.core_attention(q, k, v, attention_mask,
                                                     key_past=key_past,
                                                     value_past=value_past,
                                                     batch_valid_length=batch_valid_length,
                                                     zactivate_len=zactivate_len,
                                                     batch_index_pad=batch_index_pad,
                                                     seq_length_tensor_pad=seq_length_tensor_pad)

        # =================
        # Output. [b, s, h]
        # =================
        attention_output = self.out_proj(context)

        return attention_output, layer_present


class IFlytekSparkLinear(_Linear):
    """Linear layer with parallelism.
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: Optional[bool] = True,
                 weight_init='xavier_normal',
                 bias_init: Optional[str] = 'zeros',
                 compute_type=ms.float16,
                 transpose_b=True,
                 partition_stride: Optional[int] = 1,
                 skip_bias_add: Optional[bool] = False,):
        super(IFlytekSparkLinear, self).__init__(input_size, output_size, weight_init, bias_init, has_bias=bias,
                                                 transpose_b=transpose_b, param_init_type=compute_type,
                                                 compute_dtype=compute_type)
        # Keep input parameters
        self.compute_type = compute_type
        self.partition_stride = partition_stride
        self.skip_bias_add = skip_bias_add
        self.reshape = P.Reshape()

    def construct(self, x):
        """Forward process of linear layer"""
        out_shape = P.Shape()(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        if self.expert_flag:
            if self.use_expert_group_size is True:
                x = P.Reshape()(x, (-1, self.expert_num, self.expert_group_size, self.in_channels))
            else:
                x = P.Reshape()(x, (self.outer_batch, self.expert_num, -1, self.in_channels))

        x = self.matmul(x, self.weight)

        x = self.bias_add(x, self.bias) if self.has_bias else x

        if self.activation_flag:
            x = self.activation(x)
        output = P.Reshape()(x, out_shape)

        return output


class IFlytekSparkMLP(nn.Cell):
    """iFlytekSpark model MLP module."""
    def __init__(
            self,
            cfg,
            embed_dim: int,
            mlp_embed_dim: int,
            compute_type=ms.float16,
            skip_last_bias_add: bool = True,
            gate_gelu: bool = True
    ) -> None:
        super(IFlytekSparkMLP, self).__init__()

        self.embed_dim = embed_dim
        self.mlp_embed_dim = mlp_embed_dim
        self.compute_type = compute_type
        self.gate_gelu = gate_gelu

        weight_param0 = initializer('xavier_normal', [mlp_embed_dim // 2, embed_dim], self.compute_type)
        self.fc0 = self.build_fc0(
            init_method=weight_param0,
            # init_method=weight_param[::2, ...],
            skip_bias_add=False,
        )

        # input: dp, 1, 1 -> out: dp,1,mp
        self.fc0.shard(strategy_matmul=((cfg.parallel_config.data_parallel, 1),
                                        (cfg.parallel_config.model_parallel, 1)),
                       strategy_bias=((cfg.parallel_config.data_parallel, cfg.parallel_config.model_parallel),
                                      (cfg.parallel_config.model_parallel,)))

        weight_param1 = initializer('xavier_normal', [mlp_embed_dim // 2, embed_dim], self.compute_type)
        self.fc1 = self.build_fc1(
            init_method=weight_param1,
            skip_bias_add=False,
        )

        # input: dp, 1, 1 -> out: dp,1,mp
        self.fc1.shard(strategy_matmul=((cfg.parallel_config.data_parallel, 1),
                                        (cfg.parallel_config.model_parallel, 1)),
                       strategy_bias=((cfg.parallel_config.data_parallel, cfg.parallel_config.model_parallel),
                                      (cfg.parallel_config.model_parallel,)))
        # input: dp, 1, mp -> out: dp,1,mp
        self.mul = P.Mul().shard(((cfg.parallel_config.data_parallel, 1, cfg.parallel_config.model_parallel),
                                  (cfg.parallel_config.data_parallel, 1, cfg.parallel_config.model_parallel)))
        self.activation_fn = P.GeLU()
        # input: dp, 1, mp -> out: dp,1,mp
        self.activation_fn.shard(((cfg.parallel_config.data_parallel, 1, cfg.parallel_config.model_parallel),))

        self.fc2 = self.build_fc2(
            skip_bias_add=skip_last_bias_add,
        )
        # input: dp, 1, mp -> dp, 1, 1
        self.fc2.shard(strategy_matmul=((cfg.parallel_config.data_parallel, cfg.parallel_config.model_parallel),
                                        (1, cfg.parallel_config.model_parallel)),
                       strategy_bias=((cfg.parallel_config.data_parallel, 1),
                                      (1,)))

    def build_fc0(
            self,
            init_method='xavier_normal',
            skip_bias_add: Optional[bool] = False,
    ):
        return IFlytekSparkLinear(
            self.embed_dim,
            self.mlp_embed_dim // 2,
            compute_type=self.compute_type,
            weight_init=init_method,
            skip_bias_add=skip_bias_add,
        )

    def build_fc1(
            self,
            init_method='xavier_normal',
            skip_bias_add: Optional[bool] = False,
    ):
        """build full connection layer fc1."""
        return IFlytekSparkLinear(
            self.embed_dim,
            self.mlp_embed_dim // 2,
            compute_type=self.compute_type,
            weight_init=init_method,
            skip_bias_add=skip_bias_add,
        )

    def build_fc2(
            self,
            init_method: Optional[Callable] = 'xavier_normal',
            skip_bias_add: Optional[bool] = False,
    ):
        """build full connection layer fc2."""
        if self.gate_gelu:
            return IFlytekSparkLinear(
                self.mlp_embed_dim // 2,
                self.embed_dim,
                compute_type=self.compute_type,
                weight_init=init_method,
                skip_bias_add=skip_bias_add,
            )
        return IFlytekSparkLinear(
            self.mlp_embed_dim,
            self.embed_dim,
            compute_type=self.compute_type,
            weight_init=init_method,
            skip_bias_add=skip_bias_add,
        )

    # pylint: disable=W0221
    def construct(self, inputs):
        # gate linear
        intermediate_gate = self.fc0(inputs)
        intermediate_gate = self.activation_fn(intermediate_gate)
        intermediate_state = self.fc1(inputs)
        intermediate_parallel = self.mul(intermediate_gate, intermediate_state)
        output = self.fc2(intermediate_parallel)

        return output


class IFlytekSparkTransformerEncoderLayer(nn.Cell):
    """ iFlytekSpark transformer layer"""
    def __init__(
            self,
            config: IFlytekSparkConfig,
            rank_id: int = 0,
            layer_number: Optional[int] = None,
            self_attention: Optional[bool] = False,
            encoder_decoder_attention: Optional[bool] = False,
    ):
        assert self_attention ^ encoder_decoder_attention,\
            "self_attention and encoder_decoder_attention cannot be both True or both False"
        super(IFlytekSparkTransformerEncoderLayer, self).__init__()

        self.layer_number = layer_number
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.embed_dim = config.hidden_size
        self.mlp_embed_dim = config.ffn_hidden_size
        self.num_heads = config.num_heads
        self.kdim = config.hidden_size  # cfg.kdim
        self.vdim = config.hidden_size  # cfg.vdim
        self.hidden_dropout_p = config.dropout_rate  # cfg.hidden_dropout_p
        self.normalize_before = not config.post_layernorm_residual
        self.bias_dropout_fusion = config.bias_dropout_fusion
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.compute_type = config.compute_type  # if True else ms.float32

        self.layer_norm = _LayerNorm(
            self.embed_dim,
            eps=config.layernorm_epsilon,
            param_init_type=config.layernorm_compute_type,
        )
        self.layer_norm.shard(((config.parallel_config.data_parallel, 1, 1),))

        skip_last_bias_add = True
        self.attention = self.build_attention(config, rank_id, skip_last_bias_add)

        self.final_layer_norm = _LayerNorm(
            self.embed_dim,
            eps=config.layernorm_epsilon,
            param_init_type=config.layernorm_compute_type,
        )
        self.final_layer_norm.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.mlp = self.build_mlp(config, skip_last_bias_add)
        self.dropout = nn.Dropout(p=self.hidden_dropout_p)
        self.dropout.dropout.shard(((config.parallel_config.data_parallel, 1, 1),))

        self.add = P.Add().shard(((config.parallel_config.data_parallel, 1, 1),
                                  (config.parallel_config.data_parallel, 1, 1)))

        self.use_past = config.use_past
        self.is_lite_infer = config.is_lite_infer
        self.dtype = mstype.bfloat16 if config.compute_type == mstype.bfloat16 \
                                    else mstype.float16
        self.key_past = None
        self.value_past = None
        # only use for online infer
        if self.use_past and not self.is_lite_infer:
            print("----- [Online Infer Now] -----")
            # operator used for state reuse
            dp = config.parallel_config.data_parallel
            mp = config.parallel_config.model_parallel
            max_cache_length = config.seq_length \
                if config.seq_length <= config.sparse_local_size else config.sparse_local_size
            max_cache_batch_size = config.batch_size
            size_per_head = config.hidden_size // config.num_heads
            self.key_shape = (max_cache_batch_size, config.num_heads, max_cache_length, size_per_head)
            self.value_shape = (max_cache_batch_size, config.num_heads, max_cache_length, size_per_head)
            # parameters saving key and value states
            self.key_past = Parameter(Tensor(
                np.zeros(shape=self.key_shape), self.dtype), name="key_past", requires_grad=False)
            self.value_past = Parameter(Tensor(
                np.zeros(shape=self.value_shape), self.dtype), name="value_past", requires_grad=False)
            self.mul = P.Mul().shard(((dp, mp, 1, 1), (1,)))
            self.assign = P.Assign().shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))

        if config.seq_parallel:
            self.dropout.dropout.shard(((config.parallel_config.data_parallel,
                                         config.parallel_config.model_parallel, 1),))
            self.add = P.Add().shard(((config.parallel_config.data_parallel, config.parallel_config.model_parallel, 1),
                                      (config.parallel_config.data_parallel, config.parallel_config.model_parallel, 1)))
            self.final_layer_norm.shard(((config.parallel_config.data_parallel,
                                          config.parallel_config.model_parallel, 1),))
            self.final_layer_norm.layer_norm.add_prim_attr("recompute_comm_op", True)
            self.layer_norm.shard(((config.parallel_config.data_parallel, config.parallel_config.model_parallel, 1),))
            self.layer_norm.layer_norm.add_prim_attr("recompute_comm_op", True)
            self.attention.q_proj.reshape.add_prim_attr("recompute_comm_op", True)
            self.attention.k_proj.reshape.add_prim_attr("recompute_comm_op", True)
            self.attention.v_proj.reshape.add_prim_attr("recompute_comm_op", True)
            self.mlp.fc0.reshape.add_prim_attr("recompute_comm_op", True)
            self.mlp.fc1.reshape.add_prim_attr("recompute_comm_op", True)
            self.mlp.fc2.matmul.shard(((config.parallel_config.data_parallel, config.parallel_config.model_parallel),
                                       (1, config.parallel_config.model_parallel)),
                                      ((config.parallel_config.data_parallel *
                                        config.parallel_config.model_parallel, 1),))
            self.mlp.fc2.bias_add.shard(
                ((config.parallel_config.data_parallel * config.parallel_config.model_parallel, 1), (1,)))

    def build_attention(self, cfg, rank_id, skip_last_bias_add: bool = True):
        return IFlytekSparkMultiheadAttention(
            cfg,
            rank_id=rank_id,
            layer_number=self.layer_number,
            skip_last_bias_add=skip_last_bias_add,
        )

    def build_mlp(self, cfg, skip_last_bias_add: bool = True):
        return IFlytekSparkMLP(
            cfg,
            embed_dim=self.embed_dim,
            mlp_embed_dim=self.mlp_embed_dim,
            compute_type=self.compute_type,
            skip_last_bias_add=skip_last_bias_add,
        )

    # pylint: disable=W0221
    def construct(
            self,
            hidden_states,
            query_hidden_states: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            init_reset=True,
            batch_valid_length=None,
            zactivate_len=None,
            batch_index_pad=None,
            seq_length_tensor_pad=None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attention_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attention_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # Stage 1: MultiHeadAttention
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.layer_norm(hidden_states)
            if self.apply_residual_connection_post_layernorm:
                residual = hidden_states

        if query_hidden_states is None:
            query_hidden_states = hidden_states

        key_reset = None
        value_reset = None
        if self.use_past and self.is_first_iteration and not self.is_lite_infer:
            self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            key_reset = self.key_past
            self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            value_reset = self.value_past
            # add dependency for desired execution order
            hidden_states = F.depend(hidden_states, key_reset)
            hidden_states = F.depend(hidden_states, value_reset)
            query_hidden_states = F.depend(query_hidden_states, key_reset)
            query_hidden_states = F.depend(query_hidden_states, value_reset)

        attention_output, layer_present = self.attention(
            q=F.cast(query_hidden_states, self.compute_type),
            k=F.cast(hidden_states, self.compute_type),
            v=F.cast(hidden_states, self.compute_type),
            attention_mask=attention_mask,
            key_past=self.key_past,
            value_past=self.value_past,
            batch_valid_length=batch_valid_length,
            zactivate_len=zactivate_len,
            batch_index_pad=batch_index_pad,
            seq_length_tensor_pad=seq_length_tensor_pad
        )

        attention_output = self.dropout(attention_output)
        # hidden_states = residual + attention_output
        hidden_states = self.add(residual, attention_output)

        # Stage 2: FeedForward
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
            if self.apply_residual_connection_post_layernorm:
                residual = hidden_states

        mlp_output = self.mlp(F.cast(hidden_states, self.compute_type))

        value_update = None
        key_update = None
        if self.use_past and not self.is_lite_infer:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign(self.key_past, key_present)
            key_update = self.key_past
            self.assign(self.value_past, value_present)
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = P.depend(key_update, key_reset)
            value_update = P.depend(value_update, value_reset)

        mlp_output = P.depend(mlp_output, value_update)
        mlp_output = P.depend(mlp_output, key_update)

        mlp_output = self.dropout(mlp_output)
        # hidden_states = residual + mlp_output
        hidden_states = self.add(residual, mlp_output)

        return hidden_states


class IFlytekSparkTransformer(nn.Cell):
    """Transformer class."""

    def __init__(
            self,
            config: IFlytekSparkConfig,
            rank_id: int = 0,
    ) -> None:
        super(IFlytekSparkTransformer, self).__init__()

        # Number of layers:
        self.num_layers = config.num_layers

        # Transformer layers.
        def build_layer(layer_number, rank_id, config):
            layer = IFlytekSparkTransformerEncoderLayer(
                config,
                layer_number=layer_number,
                self_attention=True,
                encoder_decoder_attention=False,
                rank_id=rank_id,
            )
            return layer

        self.layers = nn.CellList()
        for i in range(self.num_layers):
            block = build_layer(i, rank_id=rank_id, config=config)
            set_parallel_configure_for_layer(block, i, 0, config.parallel_config, self.num_layers)
            self.layers.append(block)


        self.layernorm = _LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            param_init_type=config.layernorm_compute_type,
        )
        if config.parallel_config.pipeline_stage > 1:
            self.layernorm.set_comm_fusion(2)
        else:
            self.layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.layernorm.shard(((config.parallel_config.data_parallel, 1, 1),))
        self.layernorm.pipeline_stage = config.parallel_config.pipeline_stage - 1

    # pylint: disable=W0221
    def construct(
            self,
            hidden_states,
            attention_mask,
            init_reset=True,
            batch_valid_length=None,
            zactivate_len=None,
            batch_index_pad=None,
            seq_length_tensor_pad=None
    ):
        """Forward of transformer block"""
        for index in range(self.num_layers):
            hidden_states = self.layers[index](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                init_reset=init_reset,
                batch_valid_length=batch_valid_length,
                zactivate_len=zactivate_len,
                batch_index_pad=batch_index_pad,
                seq_length_tensor_pad=seq_length_tensor_pad
            )

        hidden_states = self.layernorm(hidden_states)

        return hidden_states
