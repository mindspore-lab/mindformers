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
from mindspore.nn.wrap.loss_scale import _grad_scale
from mindspore.ops.primitive import _primexpr
from mindspore import _checkparam as Validator
from mindspore.nn.cell import Cell

from mindformers.core.clip_grad import ClipGradNorm
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['MFTrainOneStepCell', 'MFPipelineWithLossScaleCell', 'ScaleTrainOneStepCell']

state_rescale = C.MultitypeFuncGraph("state_rescale")
_grad_scale = C.MultitypeFuncGraph("grad_scale")
get_square_sum = C.MultitypeFuncGraph("get_square_sum")
apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")
expand_dims = P.ExpandDims().add_prim_attr("grad_scale", True)
reciprocal = P.Reciprocal()


@state_rescale.register("Tensor", "Tensor", "Tensor")
def _state_rescale(scale, m, v):
    F.assign(m, F.cast(m * scale, F.dtype(m)))
    F.assign(v, F.cast(v * scale, F.dtype(v)))
    return m


@get_square_sum.register("Tensor")
def _get_square_sum(x):
    norm = P.ReduceSum(False)(F.square(F.cast(x, mstype.float32)), ())
    norm = expand_dims(norm, 0)
    return norm


@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, x):
    x_dtype = F.dtype(x)
    x = x * clip_norm / global_norm
    x = F.cast(x, x_dtype)
    return x


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return F.cast(grad, mstype.float32) * reciprocal(scale)


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


class _ClipByGlobalNorm(Cell):
    r"""
    Clips tensor values by the ratio of the sum of their norms.

    Args:
        clip_norm (Union(float, int)): The clipping ratio. Default: 1.0
        use_norm (Union(float, None)): The global norm. Default: ``None``

    Inputs:
        - **x** (Union(tuple[Tensor], list[Tensor])) - Input data to clip.

    Outputs:
        Tensor, a clipped Tensor.
    """

    def __init__(self, clip_norm=1.0, use_norm=None):
        """Initialize _ClipByGlobalNorm."""
        super(_ClipByGlobalNorm, self).__init__()
        # Add interface. This parameter is not used at present
        if use_norm is not None:
            raise ValueError(f"For '{self.cls_name}', input 'use_norm' only supports None currently, "
                             f"but got 'use_norm': {use_norm}")
        Validator.check_number("clip_norm", clip_norm, 0.0, Validator.GT, self.cls_name)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()
        self.greater_equal = P.GreaterEqual()

    def construct(self, x, scaling):
        square_sum = self.hyper_map(get_square_sum, x)
        global_norm = F.sqrt(F.addn(square_sum))
        cond = self.greater_equal(global_norm, self.clip_norm * scaling)
        global_norm = F.select(cond, global_norm, self.clip_norm * scaling)
        clip_x = self.hyper_map(F.partial(apply_global_norm, self.clip_norm * scaling, global_norm), x)
        return clip_x


@_primexpr
def _check_value(clip_norm):
    Validator.check_number("clip_norm", clip_norm, 0.0, Validator.GT, "clip_by_global_norm")


def clip_by_global_norm(x, scaling, clip_norm=1.0, use_norm=None):
    r"""
    Clips tensor values by the ratio of the sum of their norms.

    Note:
        - Input `x` should be a tuple or list of tensors. Otherwise, it will raise an error.
        - On the SEMI_AUTO_PARALLEL mode or AUTO_PARALLEL mode, if the input `x` is the gradient,
          the gradient norm values on all devices will be automatically aggregated by allreduce inserted after
          the local square sum of the gradients.

    Args:
        x (Union(tuple[Tensor], list[Tensor])): Input data to clip.
        clip_norm (Union(float, int)): The clipping ratio, it should be greater than 0. Default: ``1.0`` .
        use_norm (None): The global norm. Default: ``None`` . Currently only none is supported.

    Returns:
        tuple[Tensor], a clipped Tensor. It has the same data type as `x` and each Tensor in the output tuple is the
        same as the original input shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> x1 = np.array([[2., 3.], [1., 2.]]).astype(np.float32)
        >>> x2 = np.array([[1., 4.], [3., 1.]]).astype(np.float32)
        >>> input_x = (Tensor(x1), Tensor(x2))
        >>> out = ops.clip_by_global_norm(input_x, 1.0)
        >>> print(out)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 2.98142403e-01,  4.47213590e-01],
         [ 1.49071202e-01,  2.98142403e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 1.49071202e-01,  5.96284807e-01],
         [ 4.47213590e-01,  1.49071202e-01]]))
    """

    _check_value(clip_norm)
    clip_val = _ClipByGlobalNorm(clip_norm, use_norm)(x, scaling)
    return clip_val


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


@MindFormerRegister.register(MindFormerModuleType.WRAPPER)
class ScaleTrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, scale_sense, use_clip_grad=True, max_grad_norm=1.0,
                 move_scaling_to_adam=True, **kwargs):
        super().__init__(network, optimizer, scale_sense)
        self.move_scaling_to_adam = move_scaling_to_adam
        self.step = Parameter(Tensor(0, dtype=mstype.int32), name='step_count')
        self.ones_like = P.OnesLike()
        self.partial = P.Partial()
        self.depend = P.Depend()
        self.clip_grad_norm = ClipGradNorm(max_norm=max_grad_norm)
        self.max_grad_norm = max_grad_norm
        self.use_clip_grad = use_clip_grad
        self.parallel_config = kwargs.pop("parallel_config", None)
        self.cur_iter = Parameter(Tensor(1, dtype=mstype.int32), name="current_iterator_step")
        self.last_overflow_iter = Parameter(Tensor(0, dtype=mstype.int32), name="last_overflow_iterator_step")
        self.select = P.Select()
        self.max = P.Maximum()
        self.minimum_loss_scale = Tensor(1.0, dtype=mstype.float32)
        self.reciprocal = P.Reciprocal()
        self.less_equal = P.LessEqual()
        self.logic_and = P.LogicalAnd()
        self.logic_not = P.LogicalNot()
        self.logic_or = P.LogicalOr()
        self.const_true = Tensor(True, dtype=mstype.bool_)

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense
        status = Tensor([0] * 8, mstype.int32)

        scaling_sens_filled = self.ones_like(loss) * scaling_sens.astype(loss.dtype)
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        if not self.move_scaling_to_adam:
            grads = self.hyper_map(self.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow, should_update_mv, rate = self.process_loss_scale(scaling_sens, cond)

        if should_update_mv:
            loss = self.depend(loss, self._rescale_mv(rate))
        # if there is no overflow, do optimize
        if not overflow:
            if self.move_scaling_to_adam:
                grads = clip_by_global_norm(grads, scaling_sens, self.max_grad_norm)
                loss = self.depend(loss, self.optimizer(grads, scaling_sens, self.step))
            else:
                if self.use_clip_grad:
                    grads = self.clip_by_global_norm(grads, self.max_grad_norm)
                loss = self.depend(loss, self.optimizer(grads))
            self.step += 1
        return loss, cond, scaling_sens

    def _rescale_mv(self, rate):

        self.hyper_map(self.partial(_state_rescale, rate), self.optimizer.moments1, self.optimizer.moments2)

    def process_loss_scale(self, loss_scale, overflow):
        overflow_cond = overflow
        loss_scale_on_overflow = self.select(overflow_cond, self.max(
            loss_scale * self.reciprocal(self.loss_scaling_manager.scale_factor),
            self.minimum_loss_scale), loss_scale)
        should_inc = self.less_equal(self.loss_scaling_manager.scale_window, self.cur_iter - self.last_overflow_iter)
        last_iter_cond = self.logic_or(overflow_cond, should_inc)
        last_overflow_iter = self.select(last_iter_cond, self.cur_iter, self.last_overflow_iter)
        last_iter = F.assign(self.last_overflow_iter, last_overflow_iter)
        update_scale_cond = self.logic_and(should_inc, self.logic_not(overflow_cond))
        scale_mul_res = loss_scale_on_overflow * self.loss_scaling_manager.scale_factor
        scaled_loss_scale = self.select(update_scale_cond, scale_mul_res, loss_scale_on_overflow)
        rate = scaled_loss_scale / loss_scale
        F.assign(loss_scale, scaled_loss_scale)
        inc_cur_iter = self.cur_iter + 1
        inc_cur_iter = F.depend(inc_cur_iter, last_iter)
        F.assign(self.cur_iter, inc_cur_iter)
        return overflow, update_scale_cond, rate


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
