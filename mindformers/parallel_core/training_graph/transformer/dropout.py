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
"""Dropout"""
__all__ = ['Dropout']

import mindspore as ms
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.common.generator import default_generator
from mindspore.common.tensor import Tensor


class Dropout(nn.Cell):
    r"""
    Dropout layer for parallel training.

    Args:
        drop_prob (float): Probability of dropping an element. Default: 0.5.
    """

    def __init__(self, drop_prob: float = 0.5):
        super(Dropout, self).__init__()
        self.p = drop_prob
        self.generator_step = Tensor(1, mstype.int64)
        self.seed, self.offset = default_generator._step(self.generator_step)  # pylint: disable=protected-access
        self.dropout = ms.ops.auto_generate.DropoutExt().add_prim_attr("side_effect_hidden", True)

    def construct(self, x):
        r"""
           Input:
               x: a tensor
           Returns: a tensor
        """
        if not self.training:
            return x

        out, _ = self.dropout(input=x, p=self.p, seed=self.seed, offset=self.offset)
        return out

    def shard(self, strategy):
        self.dropout.shard((strategy, (), ()))
