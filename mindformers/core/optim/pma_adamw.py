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
"""AdamW API"""
import mindspore.ops as ops

from mindspore import _checkparam as validator
from mindspore._checkparam import GT, INC_NEITHER
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F

from mindformers.core.optim.adamw import AdamW
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

op_mul = P.Mul()
op_pow = P.Pow()
op_sqrt = P.Sqrt()
op_maximum = P.Maximum()
addcmul = P.Addcmul()
pma_adamw_opt = C.MultitypeFuncGraph("pma_adamw_opt")


@pma_adamw_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "String",
                        "Int", "Int", "Float", "Tensor", "Tensor", "Tensor",
                        "Tensor", "Tensor", "Tensor", "Bool", "Tensor")
def _update_run_op(beta1, beta2, eps, step,
                   fused_algo, interleave_step, fused_num, ema_alpha,
                   lr, weight_decay, parameters, grads, exp_avg,
                   exp_avg_sq, optim_filter, pma_weight):
    """Apply AdamW optimizer to the weight parameter."""
    op_cast = P.Cast()
    if optim_filter:
        param_fp32 = op_cast(parameters, mstype.float32)
        next_param = op_mul(param_fp32, 1 - lr * weight_decay)
        gradient_fp32 = op_cast(grads, mstype.float32)

        next_param = F.depend(next_param,
                              F.assign(exp_avg,
                                       op_mul(exp_avg, beta1) + op_mul(gradient_fp32,
                                                                       op_cast(F.tuple_to_array((1.0,)),
                                                                               mstype.float32) - beta1)))
        next_param = F.depend(next_param,
                              F.assign(exp_avg_sq, addcmul(op_mul(exp_avg_sq, beta2), gradient_fp32, gradient_fp32,
                                                           op_cast(F.tuple_to_array((1.0,)), mstype.float32) - beta2)))

        bias_correction1 = 1 - op_pow(op_cast(beta1, mstype.float32), step)
        bias_correction2 = 1 - op_pow(op_cast(beta2, mstype.float32), step)
        step_size = lr / bias_correction1

        denom = op_sqrt(exp_avg_sq / bias_correction2) + eps

        return_param = next_param - op_mul(exp_avg / denom, step_size)
        if fused_algo == 'ema' and step % interleave_step == 0 and step > 0:
            F.assign(pma_weight, (1 - ema_alpha) * pma_weight + ema_alpha * return_param)
        if fused_algo == 'sma' and step % interleave_step == 0 and step > 0:
            F.assign(pma_weight, pma_weight + return_param)
        if step % (fused_num * interleave_step) == 0 and step > 0:
            if fused_algo == 'sma':
                F.assign(pma_weight, pma_weight / fused_num)
            F.assign(return_param, pma_weight)
            F.assign(pma_weight, ops.ZerosLike()(pma_weight))

        F.assign(parameters, op_cast(return_param, F.dtype(parameters)))
        return op_cast(return_param, F.dtype(parameters))
    return op_cast(grads, F.dtype(parameters))


def _check_param_value(fused_num, interleave_step, fused_algo, ema_alpha, prim_name):
    """Check the type of inputs."""
    validator.check_value_type('fused_num', fused_num, [int], prim_name)
    validator.check_value_type("interleave_step", interleave_step, [int], prim_name)
    validator.check_string(fused_algo, ["ema", "sma"], "fused_algo", prim_name)
    validator.check_value_type("ema_alpha", ema_alpha, [float], prim_name)
    validator.check_float_range(ema_alpha, 0.0, 1.0, INC_NEITHER, "ema_alpha", prim_name)
    validator.check_int(fused_num, 0, GT, "fused_num", prim_name)
    validator.check_int(interleave_step, 0, GT, "interleave_step", prim_name)


@MindFormerRegister.register(MindFormerModuleType.OPTIMIZER)
class PmaAdamW(AdamW):
    r"""
    This is the implementation of PmAdamW.

    Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `params` is a list of `dict`, the string "params", "lr", "weight_decay", and "order_params"
            are the keys can be parsed.

            - params: Required. Parameters in current group. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in optimizer will be used. Fixed and dynamic learning rate are supported.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the optimizer will be used. It should be noted that weight
              decay can be a constant value or a Cell. It is a Cell only when dynamic weight decay is applied. Dynamic
              weight decay is similar to dynamic learning rate, users need to customize a weight decay schedule only
              with global step as input, and during training, the optimizer calls the instance of WeightDecaySchedule
              to get the weight decay value of current step.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule], optional): Default: ``1e-3``.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        betas (Union[list(float), tuple(float)], optional): The exponential decay rate for the 1st and 2nd moment
            estimations. Default: (0.9, 0.999). Each element should be in range (0.0, 1.0).

        eps (float, optional): Term added to the denominator to improve numerical stability. Default: ``1e-6``.
            Should be greater than 0.

        weight_decay (Union[float, int, Cell], optional): Weight decay (L2 penalty). Default: ``0.0``.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

        fused_num (int, optional): Only after fusing every fused_num weights,
            are they updated into the network parameters. Default: ``10``.

        interleave_step (int, optional): Fusion interval,
            take weights once every `interleave_step` for fusion. Default: ``1000``.

        fused_algo (string, optional): Fusion algorithm, supporting SMA and EMA. Default: ``ema``.

        ema_alpha (float, optional): The fusion coefficient is only effective when fused_algo=ema. Default: ``0.2``.

        swap (bool, optional): Enables swap_optimizer feature when True, offloading optimizer states to CPU instead of
            storing them on NPU. When enabled, set the environment variable `MS_DEV_RUNTIME_CONF="switch_inline:False"`.
             Default: False.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

    Raises:
        TypeError: If `fused_num` is not int.
        TypeError: If `interleave_step` is not int.
        TypeError: If `fused_algo` is not string.
        TypeError: If `ema_alpha` is not float.
        ValueError: If `fused_num` is less than 0.
        ValueError: If `interleave_step` is less than 0.
        ValueError: If `ema_alpha` is not in range (0.0, 1.0).
        ValueError: If `fused_algo` is not in ['ema', 'sma'].
    """

    def __init__(self, params, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 fused_num=10, interleave_step=1000, fused_algo='ema', ema_alpha=0.2, swap=False):
        _check_param_value(fused_num, interleave_step, fused_algo, ema_alpha, self.cls_name)

        super().__init__(
            params=params,
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            swap=swap
        )

        self.fused_num = fused_num
        self.interleave_step = interleave_step
        self.fused_algo = fused_algo
        self.ema_alpha = ema_alpha
        self.pma_weight = self.clone_state(prefix=f"pma_weight_{fused_algo}", init='zeros')

    def construct(self, gradients):
        """forward process"""
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    F.partial(pma_adamw_opt, self.beta1, self.beta2, self.eps, self.global_step,
                              self.fused_algo, self.interleave_step, self.fused_num, self.ema_alpha),
                    lr, weight_decay, self._parameters, gradients, self.exp_avg, self.exp_avg_sq, self.optim_filter,
                    self.pma_weight)
            else:
                optim_result = self.hyper_map(
                    F.partial(pma_adamw_opt, self.beta1, self.beta2, self.eps, self.global_step, self.fused_algo,
                              self.interleave_step, self.fused_num, self.ema_alpha, lr),
                    weight_decay, self._parameters, gradients, self.exp_avg, self.exp_avg_sq,
                    self.optim_filter, self.pma_weight)
        else:
            optim_result = self.hyper_map(
                F.partial(pma_adamw_opt, self.beta1, self.beta2, self.eps, self.global_step,
                          self.fused_algo, self.interleave_step, self.fused_num, self.ema_alpha, lr, weight_decay),
                self._parameters, gradients, self.exp_avg, self.exp_avg_sq, self.optim_filter, self.pma_weight)

        if self.use_parallel:
            self.broadcast_params(optim_result)

        return optim_result
