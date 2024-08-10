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
"""clip grad and scale grad"""

import mindspore.ops as ops
import mindspore._checkparam as validator


class GradAccumulator:
    '''
    Gradient accumulator.

    Args:
        micro_batch_num (int): Gradient accumulation steps.
        op (str): Operate on the result of gradient accumulation. like sum, mean. Default: "mean".

    Inputs:
        - **grads** (tuple[Tensor]) - The gradients of parameters, the shape is the same as parameters.

    Outputs:
        - Tensor, accumulated gradients, the shape and type is the same asgradients.

    Raises:
        NotImplementedError: If `op` is not mean or sum.

    Examples:
        >>> from mindformers.experimental.distri_cores.grad_handler import GradAccumulator
        >>> micro_batch_num = 2
        >>> accumulator = GradAccumulator(micro_batch_num)
        >>> grad_func = ops.value_and_grad(network, grad_position=0, weights=optimizer.parameters)
        >>> loss, grads = grad_func(input_ids, labels)
        >>> grads = accumulator(grads)
        >>> if grads is not None:
        ...     print("do optimizer")
        ...     optimizer(grads)
    '''
    def __init__(self, micro_batch_num, op="mean"):
        self.counter = 0
        validator.check_non_negative_int(micro_batch_num, "accumulate_step")
        self.accumulate_step = micro_batch_num
        if op not in ["mean", "sum"]:
            raise NotImplementedError(f"{op} is not supported in GradAccumulator yet.")
        self.is_mean_op = op == "mean"
        self.map = ops.HyperMap()
        self.has_init = False
        self.need_clear = False
        self.inner_grads = None

        self.zeroslike = ops.ZerosLike()

    def _init_inner_grads(self, param):
        return self.zeroslike(param)

    def _clear_value(self, inner_grads):
        zeros = self.zeroslike(inner_grads)
        inner_grads.assign_value(zeros)

    def _mean_value(self, inner_grads):
        inner_grads.assign_value(inner_grads / self.accumulate_step)

    def __call__(self, grads):
        if not self.has_init:
            self.inner_grads = self.map(ops.partial(self._init_inner_grads), grads)
            self.has_init = True
        if self.need_clear:
            self.map(ops.partial(self._clear_value), self.inner_grads)
            self.need_clear = False
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        self.counter += 1
        if self.counter % self.accumulate_step == 0:
            if self.is_mean_op:
                self.map(ops.partial(self._mean_value), self.inner_grads)
            self.need_clear = True
            return self.inner_grads
        return None
