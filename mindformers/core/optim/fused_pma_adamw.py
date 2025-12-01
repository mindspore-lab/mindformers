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
"""FusedPmaAdamW implementation"""
from mindspore import ops

from mindspore._checkparam import GT, INC_NEITHER
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype
from mindspore.ops.operations import Cast
from mindspore.ops.composite import MultitypeFuncGraph
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindformers.core.optim import FusedAdamW

# The compute graph of optimizer
pma_adamw_opt = MultitypeFuncGraph("pma_adamw_opt")


@pma_adamw_opt.register("Function", "Bool", "Bool", "Float", "Float", "Tensor", "Tensor",
                        "Float", "String", "Int", "Int", "Tensor", "Tensor",
                        "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _run_adamw_opt(opt, amsgrad, maximize, beta1, beta2, eps, step,
                   ema_alpha, fused_algo, interleave_step, fused_num,
                   lr, weight_decay, parameters, grads, exp_avg,
                   exp_avg_sq, optim_filter, max_exp_avg_sq, pma_weight):
    """Apply AdamW optimizer to the weight parameter."""
    op_cast = P.Cast()
    step = Cast()(step, mstype.int64)
    grads = Cast()(grads, F.dtype(parameters))
    lr = float(lr)
    weight_decay = float(weight_decay)

    if optim_filter:
        if amsgrad:
            opt(parameters, exp_avg, exp_avg_sq, max_exp_avg_sq, grads, step, lr, beta1, beta2, weight_decay, eps,
                amsgrad, maximize)
        else:
            opt(parameters, exp_avg, exp_avg_sq, exp_avg_sq, grads, step, lr, beta1, beta2, weight_decay, eps,
                amsgrad, maximize)

    if fused_algo == 'ema' and (step + 1) % interleave_step == 0 and step + 1 > 0:
        F.assign(pma_weight, op_cast((1 - ema_alpha) * pma_weight + ema_alpha * parameters, F.dtype(parameters)))
    if fused_algo == 'sma' and (step + 1) % interleave_step == 0 and step + 1 > 0:
        F.assign(pma_weight, pma_weight + parameters)
    if (step + 1) % (fused_num * interleave_step) == 0 and step + 1 > 0:
        if fused_algo == 'sma':
            F.assign(pma_weight, pma_weight / fused_num)
        F.assign(parameters, pma_weight)
        F.assign(pma_weight, ops.ZerosLike()(pma_weight))

    return True


def _check_param_value(fused_num, interleave_step, fused_algo, ema_alpha, prim_name):
    """Check the type of inputs."""
    validator.check_value_type('fused_num', fused_num, [int], prim_name)
    validator.check_value_type("interleave_step", interleave_step, [int], prim_name)
    validator.check_string(fused_algo, ["ema", "sma"], "fused_algo", prim_name)
    validator.check_value_type("ema_alpha", ema_alpha, [float], prim_name)
    validator.check_float_range(ema_alpha, 0.0, 1.0, INC_NEITHER, "ema_alpha", prim_name)
    validator.check_int(fused_num, 0, GT, "fused_num", prim_name)
    validator.check_int(interleave_step, 0, GT, "interleave_step", prim_name)


class FusedPmaAdamW(FusedAdamW):
    """
    This is the implementation of PmaAdamW that uses fused operators.

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

        amsgrad (bool, optional): If True, uses the AMSGrad variant of the Adam algorithm,
            which maintains the maximum of past squared gradients instead of an exponential average.
            This can help improve convergence in some cases. Default is ``False``.

        maximize (bool, optional): If True, the optimizer maximizes the objective function
            instead of minimizing it. This is useful in cases where the goal is to maximize
            a reward or utility function. Default is ``False``.

        fused_num (int, optional): Only after fusing every fused_num weights,
            are they updated into the network parameters. Default: ``10``.

        interleave_step (int, optional): Fusion interval,
            take weights once every `interleave_step` for fusion. Default: ``1000``.

        fused_algo (string, optional): Fusion algorithm, supporting SMA and EMA. Default: ``ema``.

        ema_alpha (float, optional): The fusion coefficient is only effective when fused_algo=ema. Default: ``0.2``.

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

    def __init__(self,
                 params,
                 learning_rate=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.0,
                 amsgrad=False,
                 maximize=False,
                 swap=False,
                 fused_num=10,
                 interleave_step=1000,
                 fused_algo='ema',
                 ema_alpha=0.2):
        _check_param_value(fused_num, interleave_step, fused_algo, ema_alpha, self.cls_name)
        super().__init__(
            params=params,
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            swap=swap
        )

        self.fused_num = fused_num
        self.interleave_step = interleave_step
        self.fused_algo = fused_algo
        self.ema_alpha = ema_alpha
        self.pma_weight = self.clone_state(prefix=f"pma_weight_{fused_algo}", init='zeros')

    def construct(self, gradients):
        """forward process"""
        grads = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    F.partial(_run_adamw_opt, self.fused_adamw_opt, self.amsgrad, self.maximize, self.beta1, self.beta2,
                              self.eps, self.global_step, self.ema_alpha, self.fused_algo,
                              self.interleave_step, self.fused_num),
                    lr, weight_decay, self._parameters, grads, self.exp_avg, self.exp_avg_sq, self.optim_filter,
                    self.max_exp_avg_sq, self.pma_weight
                )
            else:
                optim_result = self.hyper_map(
                    F.partial(_run_adamw_opt, self.fused_adamw_opt, self.amsgrad, self.maximize, self.beta1, self.beta2,
                              self.eps, self.global_step, self.ema_alpha,
                              self.fused_algo, self.interleave_step, self.fused_num, lr),
                    weight_decay, self._parameters, grads, self.exp_avg, self.exp_avg_sq, self.optim_filter,
                    self.max_exp_avg_sq, self.pma_weight
                )
        else:
            optim_result = self.hyper_map(
                F.partial(_run_adamw_opt, self.fused_adamw_opt, self.amsgrad, self.maximize, self.beta1, self.beta2,
                          self.eps, self.global_step, self.ema_alpha, self.fused_algo,
                          self.interleave_step, self.fused_num, lr, weight_decay),
                self._parameters, grads, self.exp_avg, self.exp_avg_sq, self.optim_filter,
                self.max_exp_avg_sq, self.pma_weight
            )

        return optim_result
