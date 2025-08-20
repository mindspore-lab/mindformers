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
"""FusedAdamW implementation"""
from mindspore import _checkparam as validator, Parameter, ParameterTuple, Tensor
from mindspore.common import dtype as mstype
from mindspore.ops.operations import Cast
from mindspore.ops.composite import MultitypeFuncGraph
from mindspore.ops import functional as F
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.common.initializer import initializer
from mindspore.ops import auto_generate as gen

# The compute graph of optimizer
_adamw_opt = MultitypeFuncGraph("adamw_opt")


@_adamw_opt.register("Function", "Bool", "Bool", "Float", "Float", "Tensor", "Tensor", "Tensor", "Tensor",
                     "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _run_adamw_opt(opt, amsgrad, maximize, beta1, beta2, eps, step, lr, weight_decay, parameters, grads, exp_avg,
                   exp_avg_sq, optim_filter, max_exp_avg_sq):
    """Apply AdamW optimizer to the weight parameter."""
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
    return True


def _check_param_value(betas, eps, weight_decay, prim_name):
    """Check the type of inputs."""
    if eps < 0.0:
        raise ValueError(f"Invalid epsilon value: {eps}, should be >= 0.")
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}, should be >= 0 and < 1.")
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}, should be >= 0 and < 1.")
    if weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}, should be >= 0.")

    validator.check_value_type('betas', betas, [tuple, list], prim_name)
    validator.check("betas size", len(betas), "", [2], validator.IN, prim_name)
    validator.check_value_type("betas[0]", betas[0], [float], prim_name)
    validator.check_value_type("betas[1]", betas[1], [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_value_type("weight_decay", weight_decay, [float], prim_name)


class FusedAdamW(Optimizer):
    r"""
    This is the implementation of AdamW that uses fused operators.

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

        swap (bool, optional): Enables swap_optimizer feature when True, offloading optimizer states to CPU instead of
            storing them on NPU. Default: False.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `betas[0]`, `betas[1]` or `eps` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `eps` is less than or equal to 0.
        ValueError: If `betas[0]`, `betas[1]` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.
    """

    def __init__(self,
                 params,
                 learning_rate=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.0,
                 amsgrad=False,
                 maximize=False,
                 swap=False):
        _check_param_value(betas, eps, weight_decay, self.cls_name)
        super().__init__(learning_rate, params, weight_decay=weight_decay)

        self.swap = swap
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.exp_avg = self.clone_state(prefix="exp_avg", init='zeros')
        self.exp_avg_sq = self.clone_state(prefix="exp_avg_sq", init='zeros')

        # When amsgrad=False, max_exp_avg_sq isn't actually used for calculations, but it still needs to be a valid
        # iterable (not None). We reuse exp_avg_sq here to avoid allocating extra memory.
        self.max_exp_avg_sq = self.clone_state(prefix="max_exp_avg_sq", init='zeros') if amsgrad else self.exp_avg_sq

        self.amsgrad = amsgrad
        self.maximize = maximize
        self.fused_adamw_opt = gen.AdamW()

        # Since the operator increments global_step internally, it should be initialized to -1.
        self.global_step = Parameter(Tensor([-1], mstype.int32), "global_step")

    def clone_state(self, prefix, init):
        r"""clone state
        Args:
            prefix (str): The prefix name of the parameters
            init (str): The initialization method
        """
        parameter_tuple = self.parameters
        new = []
        for old_param in parameter_tuple:
            new_state = Parameter(
                initializer(init, shape=old_param.shape, dtype=mstype.float32),
                device='CPU' if self.swap else None
            )
            new_state.param_info = old_param.param_info.clone()
            if hasattr(old_param.param_info, "cloned_obj"):
                old_param.param_info.cloned_obj.append(new_state)
            else:
                old_param.param_info.cloned_obj = [new_state]
            new_state.is_init = False
            new_state.set_data(initializer(init, shape=old_param.shape, dtype=mstype.float32))
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ParameterTuple(new)

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
                              self.eps, self.global_step),
                    lr, weight_decay, self._parameters, grads, self.exp_avg, self.exp_avg_sq, self.optim_filter,
                    self.max_exp_avg_sq
                )
            else:
                optim_result = self.hyper_map(
                    F.partial(_run_adamw_opt, self.fused_adamw_opt, self.amsgrad, self.maximize, self.beta1, self.beta2,
                              self.eps, self.global_step, lr),
                    weight_decay, self._parameters, grads, self.exp_avg, self.exp_avg_sq, self.optim_filter,
                    self.max_exp_avg_sq
                )
        else:
            optim_result = self.hyper_map(
                F.partial(_run_adamw_opt, self.fused_adamw_opt, self.amsgrad, self.maximize, self.beta1, self.beta2,
                          self.eps, self.global_step, lr, weight_decay),
                self._parameters, grads, self.exp_avg, self.exp_avg_sq, self.optim_filter, self.max_exp_avg_sq
            )

        return optim_result
