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
"""utils"""
from mindspore.ops import operations as P

from mindspore import nn, Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer, Normal, Zero

__all__ = ["get_attn_mask_func"]


class AttnMaskFill(nn.Cell):
    """Applies a mask to attention scores by filling masked positions with a specified value."""
    def __init__(self, config):
        super(AttnMaskFill, self).__init__()
        self.masked_fill = P.MaskedFill()
        self.cast = P.Cast()
        self.shard(config)

    def construct(self, attention_scores: Tensor, attention_mask, fill_value=-10000.0):
        attention_score = self.masked_fill(attention_scores,
                                           self.cast(attention_mask, mstype.bool_),
                                           Tensor(fill_value, attention_scores.dtype))
        return attention_score

    def shard(self, parallel_config):
        """sharding parameters"""
        dp = 1 if parallel_config is None else parallel_config.data_parallel
        tp = 1 if parallel_config is None else parallel_config.tensor_parallel

        self.masked_fill.shard(((dp, tp, 1, 1), (1, 1), ()))


class AttnMaskAdd(nn.Cell):
    """Adds a mask to attention scores by adding the mask values to the attention scores."""
    def __init__(self, config):
        super(AttnMaskAdd, self).__init__()
        self.add = P.Add()
        self.cast = P.Cast()
        self.shard(config)

    def construct(self, attention_scores: Tensor, attention_mask):
        attention_scores = self.add(attention_scores, self.cast(attention_mask, attention_scores.dtype))

        return attention_scores

    def shard(self, parallel_config):
        """sharding parameters"""
        dp = 1 if parallel_config is None else parallel_config.data_parallel
        tp = 1 if parallel_config is None else parallel_config.tensor_parallel
        cp = 1 if parallel_config is None else parallel_config.context_parallel

        self.add.shard(((dp, tp, cp, 1), (cp, 1)))


ATTNMASK_FUNC_MAP = {
    "attn_mask_fill": AttnMaskFill,
    "attn_mask_add": AttnMaskAdd,
}


def get_attn_mask_func(mask_func_type):
    r"""
    Get attention mask function.

    Args:
        mask_func_type (str): The attention mask function type.

    Returns:
        Function, the attention mask function.
    """
    if mask_func_type not in ATTNMASK_FUNC_MAP:
        raise KeyError("Invalid attention mask function. Supported attention "
                       "mask function are ['attn_mask_fill', 'attn_mask_add'] "
                       ", but got {}.".format(mask_func_type))
    return ATTNMASK_FUNC_MAP[mask_func_type]

def init_method_normal(sigma: float, params_dtype: mstype = mstype.float32):
    """Init method based on N(0, sigma)."""
    def init_(tensor: Tensor):
        return initializer(Normal(mean=0.0, sigma=sigma), tensor.shape, params_dtype)

    return init_


def init_method_zero(params_dtype: mstype = mstype.float32):
    """Init method based on zeros."""
    def init_(tensor: Tensor):
        return initializer(Zero(), tensor.shape, params_dtype)

    return init_
