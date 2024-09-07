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
"""Linear Layers with LoRA."""

from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore.nn.cell import Cell
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindformers.modules.layers import Linear, Dropout
from mindformers.pet.pet_config import SLoraConfig


class SLoraLinear(Cell):
    """
    Decorator of Linear layer for S-LoRA
    """
    def __init__(self, linear: Linear, adapter_ids: Tensor, config: SLoraConfig):
        super().__init__()
        self.lora_linear = linear
        self.adapter_ids = adapter_ids
        self.lora_num = config.lora_num
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_scaling = self.lora_alpha / self.lora_rank

        self.lora_mul = P.Mul()
        self.lora_add = P.Add()
        self.gather = P.Gather()
        self.cast = P.Cast()
        self.reshape = P.Reshape()

        self.lora_dropout = Dropout(keep_prob=1 - config.lora_dropout)
        self.lora_a_matmul = P.BatchMatMul(transpose_b=True)
        self.lora_b_matmul = P.BatchMatMul(transpose_b=True)

        lora_a_shape = [self.lora_num, self.lora_rank, self.lora_linear.in_channels]
        lora_b_shape = [self.lora_num, self.lora_linear.out_channels, self.lora_rank]
        self.lora_a = Parameter(initializer('zero', lora_a_shape, config.lora_dtype), requires_grad=False)
        self.lora_b = Parameter(initializer('zero', lora_b_shape, config.lora_dtype), requires_grad=False)

    def construct(self, x, expert_ids=None):
        """Forward process, x should be a tensor"""
        batch_size = self.adapter_ids.shape[0]
        out_shape = self.lora_linear.shape(x)[:-1] + (self.lora_linear.out_channels,)
        x = self.lora_linear.reshape(x, (-1, self.lora_linear.in_channels))
        if self.lora_linear.expert_flag and not self.lora_linear.use_gmm:
            if self.lora_linear.use_expert_group_size is True:
                x = self.lora_linear.reshape(x, (
                    -1, self.lora_linear.expert_num, self.lora_linear.expert_group_size, self.lora_linear.in_channels))
            else:
                x = self.lora_linear.reshape(x, (
                    self.lora_linear.outer_batch, self.lora_linear.expert_num, -1, self.lora_linear.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.lora_linear.cast(self.lora_linear.weight, self.lora_linear.dtype)
        lora_a = self.lora_linear.cast(self.lora_a, self.lora_linear.dtype)
        lora_b = self.lora_linear.cast(self.lora_b, self.lora_linear.dtype)
        lora_a = self.gather(lora_a, self.adapter_ids.reshape(-1), 0)
        lora_b = self.gather(lora_b, self.adapter_ids.reshape(-1), 0)

        x = self.lora_linear.cast(x, self.lora_linear.dtype)
        if self.lora_linear.use_gmm:
            dense_result = self.lora_linear.matmul([x], [weight], None, None, None, None, None, expert_ids)[0]
        else:
            dense_result = self.lora_linear.matmul(x, weight)

        #-------- LoRA part ----------
        x = self.reshape(x, (batch_size, -1, self.lora_linear.in_channels))

        x = self.lora_dropout(x)
        lora_result = self.lora_a_matmul(x, lora_a)
        lora_result = self.lora_b_matmul(lora_result, lora_b)
        lora_scaling = self.cast(self.lora_scaling, self.lora_linear.dtype)
        lora_result = self.reshape(lora_result, dense_result.shape)
        lora_result = self.lora_mul(lora_result, lora_scaling)

        if self.lora_linear.has_bias:
            dense_result = self.lora_linear.bias_add(dense_result, self.lora_linear.cast(
                self.lora_linear.bias, self.lora_linear.dtype))
        # Result addition
        out = self.lora_add(dense_result, lora_result)
        if self.lora_linear.activation_flag:
            out = self.lora_linear.activation(out)
        out = self.lora_linear.cast(out, ori_dtype)
        output = self.lora_linear.reshape(out, out_shape)
        return output

    def shard(self):
        r"""
         Set the shard for the linear. the strategy size should be equal to the inputs.

         Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.
         """
        strategy = self.lora_linear.matmul.in_strategy
        self.lora_a_matmul.shard(((strategy[0][0], 1, strategy[0][1]), (strategy[0][0], 1, strategy[1][1])))
        self.lora_b_matmul.shard(((strategy[0][0], 1, 1), (strategy[0][0], strategy[1][0], 1)))
        self.lora_mul.shard(((strategy[0][0], strategy[1][0]), ()))
        self.lora_add.shard(((strategy[0][0], strategy[1][0]), (strategy[0][0], strategy[1][0])))
