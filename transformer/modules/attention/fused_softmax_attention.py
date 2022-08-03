# Copyright 2022 Huawei Technologies Co., Ltd
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
"""FusedSoftmaxAttention."""
import mindspore
from mindspore import context
from mindspore.ops import DataType, CustomRegOp
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype

from mindspore.nn.transformer.transformer import default_transformer_config

# pylint: disable=C0103
FusedSoftmaxAttention = mindspore.nn.transformer.MultiHeadAttention


def func(obj, fused_kernel_path):
    """
        A wrapper of the init function of the MultiHeadAttention
        Aims to add additional fused softmax operator register
    """
    ori = obj.__init__

    def new_func(*args, **kwargs):
        ori(*args, **kwargs)
        parallel_config = kwargs.get('parallel_config', default_transformer_config)
        dp, mp = parallel_config.data_parallel, parallel_config.model_parallel
        obj.manual_fused_softmax = fused_kernel_path is not None and context.get_context("device_target").lower() in [
            "gpu"]
        if obj.manual_fused_softmax:
            fused_kernel_name = "FusedSoftMax"
            fused_grad_kernel_path = fused_kernel_path
            fused_grad_kernel_name = fused_kernel_name + "_BACK"

            fused_gpu_info = CustomRegOp() \
                .input(0, "x1") \
                .input(1, "x2") \
                .output(0, "y") \
                .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None) \
                .target("GPU").get_op_info()
            softmax_back = P.Custom(fused_grad_kernel_path + ":" + fused_grad_kernel_name,
                                    lambda x, _: x, lambda x, _: x, "aot", reg_info=fused_gpu_info)

            def bprop(_, y, out, dout):
                dx = softmax_back(out, dout)
                return dx, P.ZerosLike()(y)

            obj.softmax_fused = P.Custom(fused_kernel_path + ":" + fused_kernel_name,
                                         lambda x, _: x,
                                         lambda x, _: x,
                                         "aot",
                                         reg_info=fused_gpu_info, bprop=bprop).shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))

    return new_func


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
    # Attention score [bs, num_heads, seq_length, seq_length]
    factor = P.Cast()(self.scale_factor, P.DType()(query))
    query = self.real_div(query, factor)
    key = self.real_div(key, factor)
    score = self.batch_matmul(query, key)

    ori_dtype = P.DType()(score)
    score = P.Cast()(score, self.softmax_dtype)

    # for input size of (bs, 1) namely the second graph,
    # the shape of attention_mask matrix should be (bs, 1, 1, seq_length)
    if self.use_past and not self.is_first_iteration:
        # Calculate the current total token
        current_index = self.reducesum(F.cast(self.not_equal(self.slice(key, (0, 0, 0, 0),
                                                                        (F.shape(query)[0], 1, 1, self.seq_length),
                                                                        (1, 1, 1, 1)),
                                                             0), mstype.float32), (1, 2, 3))
        # Get the precise position index
        index = self.sub1(F.cast(current_index, mstype.int32), 1)
        index = F.reshape(index, (-1, 1, 1))
        # Calculate the attention_mask matrix via the position index
        attention_mask = F.cast(self.tensor_le(self.range, index), mstype.int32)
        attention_mask = self.expand_dims(attention_mask, 2)

    if self.manual_fused_softmax:
        multiplu_out = P.Cast()(self.sub(
            P.Cast()(F.tuple_to_array((1.0,)), P.DType()(attention_mask)),
            attention_mask), mstype.uint8)
        attention_probs = self.softmax_fused(score, multiplu_out)
    else:
        # Minus 10000 for the position where masked to exclude them from softmax
        multiplu_out = self.sub(
            P.Cast()(F.tuple_to_array((1.0,)), P.DType()(score)),
            P.Cast()(attention_mask, P.DType()(score)))

        adder = self.mul(multiplu_out, self.multiply_data)
        attention_scores = self.add(adder, score)
        # attention probs
        # pylint: disable=W0212
        attention_probs = self._softmax(attention_scores)
    attention_probs = P.Cast()(attention_probs, ori_dtype)

    attention_probs = self.prob_dropout(attention_probs)
    # Weighted sum output [bs, num_heads, seq_length, size_per_head]
    weighted_values = self.batch_matmul(attention_probs, value)
    # pylint: disable=W0212
    attention_merge = self._merge_heads(weighted_values)
    return attention_merge


def override_attention(softmax_kernel_path):
    """Replace the attention with the fused softmax"""
    FusedSoftmaxAttention.__init__ = func(FusedSoftmaxAttention, softmax_kernel_path)
    # pylint: disable=W0212
    FusedSoftmaxAttention._attn = _attn
