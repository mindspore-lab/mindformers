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
"""LLaMA transformer Layer's APIs."""
from typing import Tuple
import math
import numpy as np

try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator

from mindspore import context
from mindspore import nn, ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.context import ParallelMode
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.models.llama.llama_layer import LlamaFeedForward, LlamaRMSNorm, LlamaRotaryEmbedding
from mindformers.modules.layers import _check_past_none_input_none, _check_input_dtype, Linear
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.transformer.op_parallel_config import _check_config


class LLamaAttention(nn.Cell):
    r"""
    This is an implementation of multihead attention in LLaMA.

    Args:
            - **batch_size** (int): The batch size of the input tensor when do increnmental prediction. Should be a
                positive value.
                When do training or prediction, the argument will not work and the user can just pass None to the
                argument.
            - **src_seq_length** (int): The sequence length of the query vector.
            - **tgt_seq_length** (int): The sequence length of the key and value vector.
            - **dim** (int): The hidden size of the input.
            - **head_dim** (int): The dim of head.
            - **n_heads** (int): The number of the heads.
            - **compute_dtype** (dtype.Number): The computation type of dense. Default mstype.float16.
                Should be mstype.float32 or mstype.float16.
            - **softmax_compute_type** (dtype.Number): The type of softmax computation module. Default mstype.float32.
                Should be mstype.float32 or mstype.float16.
            - **param_init_type** (dtype.Number): The parameter initialization type of the module. Default mstype.
                float32. Should be mstype.float32 or mstype.float16.
            - **use_past** (bool): Use the past state to compute, used for incremental prediction.
                For example, if we have two words and want to generate the ten more words.
                We just need to compute the two words' state only once, and generate the next word one by one.
                When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`. At this moment,
                pass the single step's input tensor, and loop it. Default False.
            - **parallel_config** (OpParallelConfig): The parallel configure. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

    Inputs:
            - **x** (Tensor) - The input tokens with shape (batch_size, src_seq_length, hidden_size) or
                (batch_size * src_seq_length, hidden_size), if the use_past is False or is_first_iteration=True.
                Otherwise, must be (batch_size, 1, hidden_size)
            - **freqs_cis** (Tuple) - The precompute freqs and mask for rotary position embedding used in attention.
            - **attention_mask** (Tensor) - If the use_past is False or is_first_iteration=True, the attention mask
                matrix should ba (batch_size, src_seq_length, tgt_seq_length), or None. None means there will be no mask
                in softmax computation. Otherwise, the mask must be (batch_size, 1, tgt_seq_length)
            - **key_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, size_per_head, tgt_seq_length).
                The past calculated key vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **value_past** (Tensor) - Float16 tensor with shape (batch_size, num_heads, tgt_seq_length,
                size_per_head).
                The past calculated value vector. Used for incremental prediction when the use_past is True.
                Default None.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape (batch_size,) the past calculated the index.
                Used for incremental prediction when the use_past is True. Default None.

    Outputs:
            Tuple, a tuple contains(`output`, `layer_present`)

            - **output** (Tensor) - Tensor, the float tensor of the output of the layer with
                shape (batch_size, src_seq_length, hidden_size) or (batch_size * src_seq_length, hidden_size),
                if the use_past is False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size).

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
                ((batch_size, num_heads, size_per_head, tgt_seq_length),
                (batch_size, num_heads, tgt_seq_length, size_per_head)).
    """
    def __init__(self,
                 batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 head_dim,
                 dim: int = 512,
                 n_heads: int = 8,
                 compute_dtype=mstype.float16,
                 softmax_compute_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        self._is_ascend = context.get_context('device_target') in ["Ascend"]
        self.dp = parallel_config.data_parallel
        self.is_parallel_mode = _get_parallel_mode() in (
            ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if batch_size:
            Validator.check_positive_int(batch_size)
        self.apply_rotary_emb = LlamaRotaryEmbedding(head_dim, compute_dtype, parallel_config)
        self.reshape = P.Reshape()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            _check_config(parallel_config)
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.hidden_size = dim
            self.n_head = n_heads
            self.batch_size = batch_size
            if self.hidden_size % self.n_head != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                                 "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                                 .format(self.hidden_size, self.n_head))
            if self.n_head % parallel_config.model_parallel != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'n_head' must be a multiple of "
                                 "'parallel_config.model_parallel', but got the n_head is {} "
                                 "and the parallel_config.model_parallel  is {}."
                                 .format(self.n_head, parallel_config.model_parallel))
            self.is_first_iteration = True
            # Output layer
            self.wo = Linear(in_channels=self.hidden_size,
                             out_channels=self.hidden_size,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            self.wo.shard(strategy_bias=((parallel_config.data_parallel, 1), (1,)),
                          strategy_matmul=((parallel_config.data_parallel,
                                            parallel_config.model_parallel), (1, parallel_config.model_parallel)))
            self.transpose = P.Transpose()
            self.merger_head_transpose = P.Transpose()
            # embedding size per head
            self.size_per_head = self.hidden_size // self.n_head
            self.concat_k = P.Concat(axis=3)
            self.concat_v = P.Concat(axis=2)
            self.multiply_data = Tensor([
                -10000.0,
            ], dtype=compute_dtype)
            self.one = Tensor([
                1.0,
            ], dtype=compute_dtype)
            self.batch_matmul = P.BatchMatMul()
            self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
            self.real_div = P.RealDiv()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.mul_mask = P.Mul()
            self.add = P.Add()
            # Normalize factor for attention, sqrt(dk) as widely used
            self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.size_per_head))
            self.beta = Tensor(1.0)
            self.use_past = use_past
            self.softmax = nn.Softmax().to_float(softmax_compute_dtype)
            self.softmax_3d = nn.Softmax().to_float(softmax_compute_dtype)
            self.expand_dims = P.ExpandDims()

            # Query
            self.wq = Linear(self.hidden_size,
                             self.hidden_size,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            # Key
            self.wk = Linear(self.hidden_size,
                             self.hidden_size,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            # Value
            self.wv = Linear(self.hidden_size,
                             self.hidden_size,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)

            self.dtype = compute_dtype
            self.softmax_dtype = softmax_compute_dtype
            if self.use_past:
                # operators used for state reuse
                seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
                self.range = Tensor(
                    np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
                self.seq_length = src_seq_length
                self.attention_mask = Tensor(
                    np.tril(np.ones(shape=(self.seq_length, self.seq_length))), mstype.int32)
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
                self.tensor_le = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
                self.add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
                self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
                self.sub1 = P.Sub().shard(((1,), ()))
                self.tile = P.Tile().shard(((1, 1, 1, 1),))
                self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
                self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        else:
            _check_config(parallel_config)
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.hidden_size = dim
            self.n_head = n_heads
            self.batch_size = batch_size
            if self.hidden_size % self.n_head != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                                 "of 'n_head', but got the hidden_size is {} and the n_head is {}."
                                 .format(self.hidden_size, self.n_head))
            if self.n_head % parallel_config.model_parallel != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'n_head' must be a multiple of "
                                 "'parallel_config.model_parallel', but got the n_head is {} "
                                 "and the parallel_config.model_parallel  is {}."
                                 .format(self.n_head, parallel_config.model_parallel))
            self.is_first_iteration = True
            # Output layer
            self.wo = Linear(in_channels=self.hidden_size,
                             out_channels=self.hidden_size,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            self.wo.shard(strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                                           (1, parallel_config.model_parallel)))
            self.transpose = P.Transpose().shard(((parallel_config.data_parallel, 1,
                                                   parallel_config.model_parallel, 1),))
            self.merger_head_transpose = P.Transpose().shard(((parallel_config.data_parallel,
                                                               parallel_config.model_parallel, 1, 1),))
            self.n_head = n_heads
            # embedding size per head
            self.size_per_head = self.hidden_size // self.n_head
            self.concat_k = P.Concat(axis=3)
            self.concat_v = P.Concat(axis=2)
            self.multiply_data = Tensor([
                -10000.0,
            ], dtype=compute_dtype)
            self.one = Tensor([
                1.0,
            ], dtype=compute_dtype)
            self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True).shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            self.batch_matmul = P.BatchMatMul().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            self.real_div = P.RealDiv().shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), ()))
            self.sub = P.Sub().shard(((1,), (parallel_config.data_parallel, 1, 1, 1)))
            self.mul = P.Mul().shard(((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1), ()))
            self.mul_mask = P.Mul().shard(((parallel_config.data_parallel, 1, 1, 1), (1,)))
            self.add = P.Add().shard(
                ((parallel_config.data_parallel, 1, 1, 1),
                 (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
            # Normalize factor for attention, sqrt(dk) as widely used
            self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.size_per_head), dtype=compute_dtype)
            self.beta = Tensor(1.0)
            self.use_past = use_past

            self.softmax = nn.Softmax().to_float(softmax_compute_dtype)
            self.softmax.softmax.shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
            self.softmax_3d = nn.Softmax().to_float(softmax_compute_dtype)
            self.softmax_3d.softmax.shard(
                ((parallel_config.data_parallel, parallel_config.model_parallel, 1),))
            self.expand_dims = P.ExpandDims().shard(
                ((parallel_config.data_parallel, 1, 1),))

            # Query
            self.wq = Linear(self.hidden_size,
                             self.hidden_size,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            # dp,mp -> dp, 1 : dp,1 -> slice -> dp , mp * mp , 1 -> all reduce -> dp, 1
            self.wq.shard(strategy_matmul=(
                (parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)))
            # Key
            self.wk = Linear(self.hidden_size,
                             self.hidden_size,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            # dp, 1 -> dp, mp
            self.wk.shard(strategy_matmul=(
                (parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)))

            # Value
            self.wv = Linear(self.hidden_size,
                             self.hidden_size,
                             has_bias=False,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
            # dp, 1 -> dp, mp
            self.wv.shard(strategy_matmul=(
                (parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)))

            self.dtype = compute_dtype
            self.softmax_dtype = softmax_compute_dtype
            if self.use_past:
                # operators used for state reuse
                seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
                self.range = Tensor(
                    np.tile(seq_range, (batch_size, 1, 1)), mstype.int32)
                self.seq_length = src_seq_length
                self.attention_mask = Tensor(
                    np.tril(np.ones(shape=(self.seq_length, self.seq_length))), mstype.int32)
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
                self.tensor_le = P.LessEqual().shard(((1, 1, 1), (1, 1, 1)))
                self.add = P.Add().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
                self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
                self.sub1 = P.Sub().shard(((1,), ()))
                self.tile = P.Tile().shard(((1, 1, 1, 1),))
                self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
                self.mul1 = P.Mul().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

    def construct(self, x: Tensor, freqs_cis: Tuple[Tensor, Tensor], attention_mask=None,
                  key_past=None, value_past=None, batch_valid_length=None):
        """Forward process of the MultiHeadAttention"""
        self._check_inputs(x, freqs_cis, attention_mask, key_past,
                           value_past, batch_valid_length)
        batch_size = self._get_batch_size_from_input(x)
        x = self.reshape(x, (-1, x.shape[-1]))
        ori_dtype = x.dtype
        # multi head attention: query, key, value are derived from the same inputs
        query = self.wq(x).astype(self.dtype)  # dp, 1 -> dp, mp
        key = self.wk(x).astype(self.dtype)    # dp, 1 -> dp, mp
        value = self.wv(x).astype(self.dtype)  # dp, 1 -> dp, mp

        # do transpose first # dp, 1, mp, 1 -> dp, mp, 1, 1
        query = self.transpose(
            query.reshape((batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                           self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # dp, 1, mp, 1 -> dp, mp, 1, 1
        key = self.transpose(
            key.reshape((batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                         self.n_head, self.size_per_head)),
            (0, 2, 1, 3))

        query, key = self.apply_rotary_emb(query, key, freqs_cis)

        # the returned shape is [bs, n_head, seq_length, size_per_head] # dp, mp -> dp, 1, mp, 1 -> dp, mp, 1, 1
        value = self.transpose(
            self.reshape(value, (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # support input shape is [bs, seq, seq] or [bs, heads, seq, seq]
        if attention_mask is not None and attention_mask.ndim == 3:
            # expand attention mask from [bs, seq, seq] -> [bs, 1, seq, seq]
            attention_mask = self.expand_dims(attention_mask, 1)
        # key and value for current token(s)
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = (
                    self.less(self.range, batch_valid_length.view(-1, 1, 1))).astype(self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = self.mul1(
                    key, self.expand_dims(valid_length_vector, 2))
                value_present = self.mul1(
                    value, self.expand_dims(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            else:
                # Get the current token position index
                valid_length = self.reducesum((self.not_equal(self.slice(key_past, (0, 0, 0, 0),
                                                                         (key.shape[0], 1, 1,
                                                                          self.src_seq_length),
                                                                         (1, 1, 1, 1)),
                                                              0)).astype(mstype.float32), (1, 2, 3))
                valid_length = self.reshape(valid_length, (-1, 1, 1))
                valid_length_vector = (self.equal(
                    valid_length, self.range)).astype(self.dtype)
                # Pad the key and value to seq_length with only the position index not zero
                current_key = self.mul1(self.tile(key, (1, 1, 1, self.seq_length)),
                                        self.expand_dims(valid_length_vector, 2))
                current_value = self.mul1(self.tile(value, (1, 1, self.seq_length, 1)),
                                          self.expand_dims(valid_length_vector, 3))
                # Concat the previous saved state and current state
                key = self.add(key_past, current_key)
                value = self.add(value_past, current_value)
                # Update key_present and value_present for state update
                key_present = key
                value_present = value
                attention_mask = self.reshape(self.attention_mask,
                                              (self.seq_length, self.seq_length, 1, 1))

        layer_present = (key_present, value_present)
        # multi head attention considering attention mask
        # the return shape is [bs * seq_length, hidden_size]
        attention = self._attn(query, key, value, attention_mask)

        # Output
        output = self.wo(attention)
        # output = self.reshape(output, ori_shape)
        output = output.astype(ori_dtype)
        return output, layer_present

    def _get_batch_size_from_input(self, input_tensor):
        """Get the batch size from query tensor"""
        # For the incremental prediction, the seq length for the input is 1.
        if input_tensor.ndim == 2 and ((self.use_past and self.is_first_iteration) or (not self.use_past)):
            return input_tensor.shape[0] // self.src_seq_length
        return input_tensor.shape[0]

    def _get_seq_length_under_incremental(self, length):
        r"""Return the length of the tensor.
            For the incremental prediction, the seq length for the input is 1.
        """
        if self.use_past and not self.is_first_iteration:
            return 1
        return length

    def _check_inputs(self, x, freqs_cis, attention_mask, key_past=None, value_past=None, batch_valid_length=None):
        r"""Check inputs"""
        freqs_cos, freqs_sin, minus_mask, rotary_mask = freqs_cis
        _check_input_dtype(x.dtype, "x", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(freqs_cos.dtype, "freqs_cos", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(freqs_sin.dtype, "freqs_sin", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(minus_mask.dtype, "mins_mask", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(rotary_mask.dtype, "rotary_mask", [mstype.float32, mstype.float16], self.cls_name)
        if attention_mask is not None:
            _check_input_dtype(attention_mask.dtype, "attention_mask", [mstype.float32, mstype.float16], self.cls_name)

        key_is_tensor = isinstance(key_past, Tensor)
        value_is_tensor = isinstance(value_past, Tensor)
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        key_is_default = key_past is None
        value_is_default = value_past is None
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "key_past", self.cls_name, None, key_is_tensor, key_is_default)
        _check_past_none_input_none(self.use_past, "value_past", self.cls_name, None, value_is_tensor, value_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)
        if self.use_past:
            _check_input_dtype(key_past.dtype, "key_past", [mstype.float16], self.cls_name)
            _check_input_dtype(value_past.dtype, "value_past", [mstype.float16], self.cls_name)
            _check_input_dtype(batch_valid_length.dtype, "batch_valid_length", [mstype.int32], self.cls_name)
        return True

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        # dp,mp,1,1 -> dp,1,mp,1
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = x.shape
        new_shape = (x_shape[0], x_shape[1], -1)
        # new_shape = (-1, x_shape[-2] * x_shape[-1])
        # dp,1，mp,1 -> dp,mp
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """
        attention_probs = self.softmax(attention_scores)
        return attention_probs

    def _attn(self, query, key, value, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # Normalize query and key before MatMul, default off
        # Attention score [bs, n_head, seq_length, seq_length] query, key, value : dp, mp, 1, 1
        score = self.batch_matmul_q_k(query, key)
        # score : b,num_head,t,t; dp, mp, 1, 1
        score = self.mul(score, self.inv_norm_factor)

        # for input size of (bs, 1) namely the second graph,
        # the shape of attention_mask matrix should be (bs, 1, 1, seq_length)
        if attention_mask is not None:
            if self.use_past and not self.is_first_iteration:
                # Calculate the current total token
                current_index = self.reducesum((self.not_equal(self.slice(key, (0, 0, 0, 0),
                                                                          (query.shape[0], 1, 1,
                                                                           self.seq_length),
                                                                          (1, 1, 1, 1)),
                                                               0)).astype(mstype.float32), (1, 2, 3))
                # Get the precise position index
                index = self.sub1(current_index.astype(mstype.int32), 1)
                index = self.reshape(index, (-1, 1, 1))
                # Calculate the attention_mask matrix via the position index
                attention_mask = (self.tensor_le(
                    self.range, index)).astype(mstype.int32)
                attention_mask = self.expand_dims(attention_mask, 2)
            # Minus 10000 for the position where masked to exclude them from softmax
            multiplu_out = self.sub(self.one, attention_mask.astype(self.dtype))  # dp,1,1,1->dp,1,1,1

            # dp,1,1,1->dp,1,1,1
            adder = self.mul_mask(multiplu_out, self.multiply_data)
            score = self.add(adder, score)  # dp,1,1,1->dp,mp,1,1

        # attention probs
        attention_probs = self._softmax(score.astype(self.softmax_dtype))

        # Weighted sum output [bs, n_head, seq_length, size_per_head]
        weighted_values = self.batch_matmul(attention_probs.astype(self.dtype), value)
        # dp,mp,1,1 -> dp,1,mp,1 -> dp,mp
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge


class LLamaDecodeLayer(nn.Cell):
    r"""
        Transformer Layer. This is an implementation of the single layer of the transformer
        encoder layer, including multihead attention and feedward layer.

        Args:
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            seq_length(int): The input sequence length.
            layer_id(int): The layer id of current transformer block layer.
            dim(int): The hidden size of the input.
            num_heads(int): The number of the heads.
            multiple_of(int): The SwiGLU hidden layer size multiple of large power of 2.
            norm_eps (float): The epsilon value of the denominator. Default 1e-5.
            compute_dtype(dtype.Number): The computation type of the layer.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            layernorm_compute_type(dtype.Number): The computation type of the norm.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            use_past(bool): Use the past state to compute, used for incremental prediction. For example, if we have two
                words and want to generate the ten more words. We just need to compute the two words' state only once,
                and generate the next word one by one. When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`.
                At this moment, pass the single step's input tensor, and loop it. Default False.
            parallel_config(OpParallelConfig, MoEParallelConfig): The parallel configure. When MoE is applied,
                MoEParallelConfig is effective, otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **x** (Tensor) - Float Tensor, shape should be [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size], if the use_past is False or is_first_iteration=True. Otherwise,
              should be [batch_size, 1, hidden_size]
            - **freqs_cis** (Tuple) - The precompute freqs and mask for rotary position embedding used in attention.
            - **input_mask** (Tensor) - Float Tensor, If the use_past is False or is_first_iteration=True,
              the attention mask matrix should ba [batch_size, seq_length, seq_length], or None. None means there will
              be no mask in softmax computation. Otherwise, should be [batch_size, 1, hidden_size]
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`).

            - **output** (Tensor) - The float tensor of the output of the layer with
              shape (batch_size, seq_length, hidden_size) or (batch_size * seq_length, hidden_size), if the use_past is
              False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size)

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
              ((batch_size, num_heads, size_per_head, seq_length),
              (batch_size, num_heads, seq_length, size_per_head)).

    """
    def __init__(self,
                 batch_size,
                 seq_length,
                 layer_id,
                 dim: int = 512,
                 n_heads: int = 8,
                 multiple_of: int = 256,
                 norm_eps: float = 1e-5,
                 compute_dtype=mstype.float16,
                 layernorm_compute_dtype=mstype.float32,
                 softmax_compute_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 parallel_config=TransformerOpParallelConfig()):
        super().__init__()
        if batch_size or use_past:
            Validator.check_positive_int(batch_size)
        self.batch_size = batch_size
        self.use_past = use_past
        self.seq_length = seq_length
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = self.hidden_size // self.n_head
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            _check_config(parallel_config)
            if self.n_head % parallel_config.model_parallel != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'n_head' must be a multiple of "
                                 "'parallel_config.model_parallel', but got the n_head is {} "
                                 "and the parallel_config.model_parallel  is {}."
                                 .format(self.n_head, parallel_config.model_parallel))
            if self.hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
                    " model_parallel is {}.".format(self.hidden_size, parallel_config.model_parallel))
            self.attention_norm = LlamaRMSNorm(self.hidden_size, norm_eps,
                                               param_init_type=param_init_type).to_float(layernorm_compute_dtype)
                                        #   parallel_config=parallel_config).to_float(layernorm_compute_dtype)
            self.ffn_norm = LlamaRMSNorm(self.hidden_size, norm_eps,
                                         param_init_type=param_init_type).to_float(layernorm_compute_dtype)
                                    # parallel_config=parallel_config).to_float(layernorm_compute_dtype)

            self.attention = LLamaAttention(batch_size=batch_size,
                                            src_seq_length=seq_length,
                                            tgt_seq_length=seq_length,
                                            head_dim=self.head_dim,
                                            dim=dim,
                                            n_heads=n_heads,
                                            compute_dtype=compute_dtype,
                                            softmax_compute_dtype=softmax_compute_dtype,
                                            param_init_type=param_init_type,
                                            use_past=use_past,
                                            parallel_config=parallel_config)
            self.feed_forward = LlamaFeedForward(dim=self.hidden_size,
                                                 hidden_dim=4*self.hidden_size,
                                                 multiple_of=multiple_of,
                                                 compute_dtype=compute_dtype,
                                                 param_init_type=param_init_type,
                                                 parallel_config=parallel_config)
            self.add = P.Add().shard(((parallel_config.data_parallel, 1),
                                      (parallel_config.data_parallel, 1)))
            self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1),
                                         (parallel_config.data_parallel, 1, 1)))
            self.dtype = compute_dtype
            self.key_past = None
            self.value_past = None
            self.reshape = P.Reshape()

            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                size_per_head = self.hidden_size // self.n_head
                self.key_shape = (batch_size, self.n_head,
                                  size_per_head, seq_length)
                self.value_shape = (batch_size, self.n_head,
                                    seq_length, size_per_head)
                # parameters saving key and value states
                self.key_past = Parameter(
                    Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
                self.value_past = Parameter(
                    Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
                self.tile = P.Tile().shard(((1, 1),))
                self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
                self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        elif _get_parallel_mode() not in (ParallelMode.AUTO_PARALLEL,):
            _check_config(parallel_config)
            if self.n_head % parallel_config.model_parallel != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'n_head' must be a multiple of "
                                 "'parallel_config.model_parallel', but got the n_head is {} "
                                 "and the parallel_config.model_parallel  is {}."
                                 .format(self.n_head, parallel_config.model_parallel))
            if self.hidden_size % parallel_config.model_parallel != 0:
                raise ValueError(
                    "For 'TransformerEncoderLayer', the class variable 'hidden_size' must be divisibled by "
                    "the 'parallel_config.model_parallel', but got the hidden_size is {} and parallel_config."
                    " model_parallel is {}.".format(self.hidden_size, parallel_config.model_parallel))
            self.attention_norm = LlamaRMSNorm(self.hidden_size,
                                               norm_eps,
                                               param_init_type=param_init_type).to_float(layernorm_compute_dtype)
            self.attention_norm.shard(((parallel_config.data_parallel, 1, 1),))
            self.ffn_norm = LlamaRMSNorm(self.hidden_size,
                                         norm_eps,
                                         param_init_type=param_init_type).to_float(layernorm_compute_dtype)
            self.ffn_norm.shard(((parallel_config.data_parallel, 1, 1),))

            self.attention = LLamaAttention(batch_size=batch_size,
                                            src_seq_length=seq_length,
                                            tgt_seq_length=seq_length,
                                            head_dim=self.head_dim,
                                            dim=dim,
                                            n_heads=n_heads,
                                            compute_dtype=compute_dtype,
                                            softmax_compute_dtype=softmax_compute_dtype,
                                            param_init_type=param_init_type,
                                            use_past=use_past,
                                            parallel_config=parallel_config)
            self.feed_forward = LlamaFeedForward(dim=self.hidden_size,
                                                 hidden_dim=4 * self.hidden_size,
                                                 multiple_of=multiple_of,
                                                 compute_dtype=compute_dtype,
                                                 param_init_type=param_init_type,
                                                 parallel_config=parallel_config)
            self.add = P.Add().shard(((parallel_config.data_parallel, 1),
                                      (parallel_config.data_parallel, 1)))
            self.add_3d = P.Add().shard(((parallel_config.data_parallel, 1, 1),
                                         (parallel_config.data_parallel, 1, 1)))
            self.dtype = compute_dtype
            self.key_past = None
            self.value_past = None
            self.reshape = P.Reshape()

            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
                self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
                self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
                size_per_head = self.hidden_size // self.n_head
                self.key_shape = (batch_size, self.n_head,
                                  size_per_head, seq_length)
                self.value_shape = (batch_size, self.n_head,
                                    seq_length, size_per_head)
                # parameters saving key and value states
                self.key_past = Parameter(
                    Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
                self.value_past = Parameter(
                    Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
                self.tile = P.Tile().shard(((1, 1),))
                self.mul = P.Mul().shard(((1, 1, 1, 1), (1,)))
                self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        else:
            raise RuntimeError(f"The {self.cls_name} only support sharding propagation or "
                               f"semi-auto parallel mode now.")

    def construct(self, x, freqs_cis, input_mask=None, init_reset=True, batch_valid_length=None):
        """ Forward of transformer block. """
        self._check_input(x, freqs_cis, input_mask,
                          init_reset, batch_valid_length)
        # dp, 1, 1 -> dp, 1, 1
        input_x = self.attention_norm(x)
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            self.assign(self.key_past, self.mul(
                self.key_past, init_reset.astype(self.dtype)))
            key_reset = self.key_past
            self.assign(self.value_past, self.mul(
                self.value_past, init_reset.astype(self.dtype)))
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = ops.depend(input_x, key_reset)
            input_x = ops.depend(input_x, value_reset)

        # dp, 1, 1 -> dp, 1, 1
        h, layer_present = self.attention(input_x, freqs_cis, input_mask,
                                          self.key_past, self.value_past, batch_valid_length)
        h = self.add_3d(x, h)
        # dp, 1, 1 -> dp, 1, 1
        ffn_norm = self.ffn_norm(h)
        # dp, 1, 1 -> dp, 1, 1
        ffn_out = self.feed_forward(ffn_norm)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign(self.key_past, key_present)
            key_update = self.key_past
            self.assign(self.value_past, value_present)
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = ops.depend(key_update, key_reset)
            value_update = ops.depend(value_update, value_reset)

        # add dependency for desired execution order
        ffn_out = ops.depend(ffn_out, value_update)
        ffn_out = ops.depend(ffn_out, key_update)
        # if shape is 3d, we reshape the inputs of the add
        out = self.add_3d(h, ffn_out)
        return out, layer_present

    def _check_input(self, x, freqs_cis, input_mask, init_reset, batch_valid_length):
        r"""Check inputs"""
        _check_input_dtype(
            x.dtype, "x", [mstype.float32, mstype.float16], self.cls_name)
        freqs_cos, freqs_sin, mins_mask, rotary_mask = freqs_cis
        _check_input_dtype(freqs_cos.dtype, "freqs_cos", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(freqs_sin.dtype, "freqs_sin", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(mins_mask.dtype, "mins_mask", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(rotary_mask.dtype, "rotary_mask", [mstype.float32, mstype.float16], self.cls_name)
        if input_mask is not None:
            _check_input_dtype(input_mask.dtype, "input_mask", [mstype.float32, mstype.float16], self.cls_name)

        init_reset_is_tensor = isinstance(init_reset, Tensor)
        init_reset_is_default = init_reset is True
        batch_valid_length_is_tensor = isinstance(batch_valid_length, Tensor)
        batch_is_default = batch_valid_length is None
        _check_past_none_input_none(self.use_past, "init_reset", self.cls_name, True, init_reset_is_tensor,
                                    init_reset_is_default)
        _check_past_none_input_none(self.use_past, "batch_valid_length", self.cls_name, None,
                                    batch_valid_length_is_tensor, batch_is_default)

        if self.use_past:
            _check_input_dtype(init_reset.dtype, "init_reset", [mstype.bool_], self.cls_name)
            _check_input_dtype(batch_valid_length.dtype, "batch_valid_length", [mstype.int32], self.cls_name)
        return True
