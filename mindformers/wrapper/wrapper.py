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
from mindspore.common.tensor import Tensor
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import nn, Parameter, ParallelMode
from mindspore.parallel._utils import _get_enable_parallel_optimizer
import mindspore.common.dtype as mstype

from mindformers.core.clip_grad import ClipGradNorm
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


__all__ = ['TrainOneStepWithClipGN', 'MFTrainOneStepCell', 'MFPipelineWithLossScaleCell']


_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return F.cast(grad, mstype.float32) * reciprocal(scale)


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

    def __init__(self,
                 network,
                 optimizer,
                 use_clip_grad=False,
                 max_grad_norm=1.0,
                 scale_sense=1.0,
                 **kwargs):
        super(MFTrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.use_clip_grad = use_clip_grad
        self.clip_grad_norm = ClipGradNorm(max_norm=max_grad_norm)
        self.parallel_config = kwargs.pop("parallel_config", None)

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


grad_scale = C.MultitypeFuncGraph("grad_scale")
shard_grad_scale = C.MultitypeFuncGraph("shard_grad_scale")


@grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_grad_scale_pipeline(scale, grad, accu_grad):
    accu_grad = F.depend(accu_grad, grad)
    new_grad = accu_grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    zeros = F.tensor_mul(accu_grad, 0.0)
    new_grad = F.depend(new_grad, F.assign(accu_grad, zeros))
    return new_grad


@shard_grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_shard_grad_scale_pipeline(scale, grad, accu_grad):
    new_grad = grad * reciprocal(scale)
    accu_grad = F.depend(accu_grad, new_grad)
    new_grad = F.depend(new_grad, F.assign(accu_grad, F.zeros_like(accu_grad)))
    return new_grad


@MindFormerRegister.register(MindFormerModuleType.WRAPPER)
class MFPipelineWithLossScaleCell(nn.TrainOneStepCell):
    r"""
    Append an train one step cell with loss scale of pipeline parallel for MindFormers.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        use_clip_grad (bool): Whether to use gradient clipping. Default: True.
        max_grad_norm (float): Maximum gradient constraint value. Default: 1.0.
        scale_sense (Cell): Cell to do the loss scale. Default: 1.0.
        micro_batch_num (int): Micro batch number of pipeline parallel. Default: 1.

    Inputs:
        - **(\*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scale value.

        - **loss** (Tensor) -  A scalar, the loss value.
        - **overflow** (Tensor) -  A scalar, whether overflow occur or not, the type is bool.
        - **loss scale** (Tensor) -  The loss scale value, the shape is :math:`()` or :math:`(1,)`.

    Raises:
        TypeError: If `scale_sense` is neither Cell nor Tensor.
        ValueError: If shape of `scale_sense` is neither (1,) nor ().
    """

    def __init__(self, network, optimizer, use_clip_grad=True, max_grad_norm=1.0,
                 scale_sense=1.0, micro_batch_num=1, **kwargs):
        super(MFPipelineWithLossScaleCell, self).__init__(network, optimizer, sens=None)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.grad_reducer = F.identity
        self.degree = 1
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        if self.parallel_mode not in [ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL]:
            raise ValueError(f"ParallelMode must be one of "
                             f"[ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL], but found "
                             f"{self.parallel_mode}.")
        self.allreduce = P.AllReduce()
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.reshape = P.Reshape()
        self.loss_scaling_manager = None
        if isinstance(scale_sense, nn.Cell):
            self.loss_scaling_manager = scale_sense
            self.scale_sense = Parameter(Tensor(scale_sense.get_loss_scale(), dtype=mstype.float32),
                                         name="scale_sense")
        elif isinstance(scale_sense, Tensor):
            if scale_sense.shape == (1,) or scale_sense.shape == ():
                self.scale_sense = Parameter(scale_sense, name='scale_sense')
            else:
                raise ValueError("The shape of 'scale_sense' must be (1,) or (), but got {}"
                                 .format(scale_sense.shape))
        else:
            raise TypeError("The 'scale_sense' must be Cell or Tensor, but got {}".format(type(scale_sense)))
        self.opt_shard = _get_enable_parallel_optimizer()
        self.use_clip_grad = use_clip_grad
        self.clip_grad_norm = ClipGradNorm(max_norm=max_grad_norm)
        self.micro_size = micro_batch_num
        self.parallel_config = kwargs.pop("parallel_config", None)

    @C.add_flags(has_effect=True)
    def construct(self, *inputs):
        """The construct processes of pipeline wrapper cell."""
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))

        init = self.alloc_status()
        status_clear = self.clear_before_grad(init)
        scaling_sens_filled = F.depend(scaling_sens_filled, status_clear)
        grads = self.grad(self.network, self.weights)(*inputs,
                                                      self.cast(scaling_sens_filled / self.micro_size,
                                                                mstype.float32))
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        loss = F.depend(loss, status_clear)

        if self.opt_shard:
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(F.partial(shard_grad_scale, scaling_sens * self.degree), grads, self.accu_grads)
        else:
            accu_grads = self.grad_reducer(self.accu_grads)
            grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads, accu_grads)

        if self.use_clip_grad:
            grads, _ = self.clip_grad_norm(grads)

        # sum overflow flag over devices
        flag_reduce = self.allreduce(flag_sum)
        cond = self.less_equal(self.base, flag_reduce)

        overflow = cond
        if self.loss_scaling_manager is not None:
            overflow = self.loss_scaling_manager(self.scale_sense, cond)

        if not overflow:
            loss = F.depend(loss, self.optimizer(grads))

        return loss, overflow, scaling_sens.value()
