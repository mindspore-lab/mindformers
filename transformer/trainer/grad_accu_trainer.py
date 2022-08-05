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
"""Used for gradient accumulation."""
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell
from mindspore import context
from mindspore import Parameter
from mindspore.common.initializer import initializer

from transformer.utils import clone_state

__all__ = ["TrainAccuStepsWithLossScaleCell"]

grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


cast = P.Cast()
update_accu_grads = C.MultitypeFuncGraph("update_accu_grads")


@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign_add(accu_grad, cast(grad, F.dtype(accu_grad))))

zeroslike = P.ZerosLike()

reset_accu_grads = C.MultitypeFuncGraph("reset_accu_grads")
update_pipeline_grads = C.MultitypeFuncGraph("update_pipeline_grads")


@update_pipeline_grads.register("Tensor", "Tensor")
def _update_pipeline_grads(accu_grad, grad):
    return F.depend(accu_grad, grad)


@reset_accu_grads.register("Tensor")
def _reset_accu_grads(accu_grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, zeroslike(accu_grad)))


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


class TrainAccuStepsWithLossScaleCell(TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph. To mimic higher batch size, gradients are
    accumulated N times before weight update.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(TrainAccuStepsWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.network = network
        self.accumulation = False
        self.accumulation_steps = context.get_auto_parallel_context("grad_accumulation_step")
        self.one = Tensor(np.array([1]).astype(np.int32))
        self.zero = Tensor(np.array([0]).astype(np.int32))
        self.accu_grads = clone_state(self.weights, prefix="accu_grads", init="zeros", is_follow=True)
        self.accumulation_scale = Tensor(np.array([self.accumulation_steps])).astype(np.int32)
        self.accu_overflow = Parameter(initializer(0, [1], mstype.int32))
        self.accu_loss = Parameter(initializer(0, [1], mstype.float32))
        self.cast = P.Cast()
        self.logical_or = P.LogicalOr()
        self.not_equal = P.NotEqual()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.hyper_map = C.HyperMap()

    def _detect_overflow_last_step(self, *inputs):
        """Check the overflow at the last step, as the GPU will check nan for each gradient"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        # alloc status and clear should be right before grad operation
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network,
                          weights)(*inputs,
                                   scaling_sens_filled)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        if self.accumulation and self.accumulation_steps > 1:
            accu_succ = self.hyper_map(update_accu_grads, self.accu_grads, grads)
            loss = F.depend(loss, accu_succ)
        ret = None
        if not self.accumulation:
            grads = self.grad_reducer(grads)

            overflow = self.get_overflow_status(status, grads)
            accu_overflow = self.select(overflow, self.one, self.zero)
            scaling = self.accumulation_scale
            grads = self.hyper_map(
                F.partial(grad_scale, scaling), grads)
            accu_overflow = self.allreduce(accu_overflow)

            overflow = self.less_equal(self.base, accu_overflow)
            accu_grads = F.depend(self.accu_grads, grads)

            accu_succ = self.hyper_map(reset_accu_grads, accu_grads)
            overflow = F.depend(overflow, accu_succ)

            overflow = self.reshape(overflow, (()))
            overflow = self.process_loss_scale(overflow)

            if not overflow:
                self.optimizer(grads)

            ret = (loss, overflow, scaling_sens)
        else:
            ret = (loss, False, scaling_sens)

        return ret

    def _detect_overflow_every_step(self, *inputs):
        """Check the overflow at the every step, as the NPU will check overflow flag for each gradient"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        # alloc status and clear should be right before grad operation
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network,
                          weights)(*inputs,
                                   scaling_sens_filled)

        if self.accumulation and self.accumulation_steps > 1:
            accu_succ = self.hyper_map(update_accu_grads, self.accu_grads, grads)
            loss = F.depend(loss, accu_succ)

        overflow = self.get_overflow_status(status, grads)
        overflow = self.logical_or(self.not_equal(self.accu_overflow, self.zero), overflow)
        accu_overflow = self.select(overflow, self.one, self.zero)

        if self.accumulation:
            self.accu_overflow = accu_overflow
        else:
            my_zero = F.depend(self.zero, accu_overflow)
            initialize = P.Assign()(self.accu_overflow, my_zero)
            grads1 = F.depend(grads, initialize)

            # apply grad reducer on grads
            grads = self.grad_reducer(grads1)
            scaling = scaling_sens * self.accumulation_steps
            grads = self.hyper_map(
                F.partial(grad_scale, scaling), grads)
            accu_overflow = self.allreduce(accu_overflow)

            overflow = self.less_equal(self.base, accu_overflow)
            accu_grads = F.depend(self.accu_grads, grads)

            accu_succ = self.hyper_map(reset_accu_grads, accu_grads)
            overflow = F.depend(overflow, accu_succ)

            overflow = self.reshape(overflow, (()))
            overflow = self.process_loss_scale(overflow)

            if not overflow:
                self.optimizer(grads)

        ret = (loss, overflow, scaling_sens)
        return ret

    def construct(self, *inputs):
        """Defines the computation performed."""
        if self.gpu_target:
            ret = self._detect_overflow_last_step(*inputs)
        else:
            ret = self._detect_overflow_every_step(*inputs)

        return ret
