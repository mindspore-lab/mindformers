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
"""Self-Define Wrapper."""

from mindspore import nn
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindformers.core.clip_grad import ClipGradNorm
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


__all__ = ['TrainOneStepWithClipGN', 'MFTrainOneStepCell']


_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


@MindFormerRegister.register(MindFormerModuleType.WRAPPER)
class TrainOneStepWithClipGN(nn.TrainOneStepWithLossScaleCell):
    """TrainOneStep"""

    def __init__(self, network, optimizer,
                 use_clip_grad=False, clip_norm=1.0,
                 scale_sense=1.0):
        super(TrainOneStepWithClipGN, self).__init__(network, optimizer, scale_sense)
        self.print = P.Print()
        self.clip_norm = clip_norm
        self.use_clip_grad = use_clip_grad

    def construct(self, *inputs):
        """construct"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.use_clip_grad:
                grads = C.clip_by_global_norm(grads, clip_norm=self.clip_norm)
            loss = F.depend(loss, self.optimizer(grads))
        else:
            self.print("==========Overflow Now============")
        return loss


@MindFormerRegister.register(MindFormerModuleType.WRAPPER)
class MFTrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    r"""TrainOneStep For MindFormer.
    Network training with loss scaling, grad clip, gradient accumulation, exponential moving average and so on.

    This is a training step with loss scaling. It takes a network, an optimizer and a scale update Cell(or a Tensor) as
    args. The loss scale value can be updated in both host side or device side. If you want to update it on
    host side, using a value of Tensor type as `scale_sense`, otherwise, using a Cell instance for updating loss
    scale as `scale_sense`.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Cell): Optimizer for updating the network parameters.
        use_clip_grad (bool): Whether to use the gradient clipping function. Default: False.
        max_grad_norm (float): Maximum gradient value. Default: 1.0.
        scale_sense (Union[Tensor, Cell]): If this value is a Cell, it will be called by `MFTrainOneStepCell`
            to update loss scale. If this value is a Tensor, the loss scale can be modified by `set_sense_scale`,
            the shape should be :math:`()` or :math:`(1,)`.

    Inputs:
        - **(*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scale value.

        - **loss** (Tensor) -  A scalar, the loss value.
        - **overflow** (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        - **loss scale** (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.

    Raises:
        TypeError: If `scale_sense` is neither Cell nor Tensor.
        ValueError: If shape of `scale_sense` is neither (1,) nor ().
    """

    def __init__(self, network, optimizer,
                 use_clip_grad=False, max_grad_norm=1.0,
                 scale_sense=1.0):
        super(MFTrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.use_clip_grad = use_clip_grad
        self.clip_grad_norm = ClipGradNorm(max_norm=max_grad_norm)

    def construct(self, *inputs):
        """forward and backward."""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.use_clip_grad:
                grads, _ = self.clip_grad_norm(grads)
            loss = F.depend(loss, self.optimizer(grads))
        return loss, overflow, scaling_sens
