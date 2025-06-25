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
"""MindFormers Optimizer."""
from typing import Union, Optional, Iterable

from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .build_optim import build_optim
from .came import Came
from .optim import (
    FP32StateAdamWeightDecay,
    FusedAdamWeightDecay
)
from .adamw import AdamW as BasicAdamW
from .fused_adamw import FusedAdamW

__all__ = ['AdamW']


@MindFormerRegister.register(MindFormerModuleType.OPTIMIZER)
class AdamW:
    r"""
    This is the implementation of AdamW.

    .. math::
        \begin{array}{l}
            &\newline
            &\hline \\
            &\textbf{Parameters}: \: 1^{\text {st }}\text {moment vector} \: m , \: 2^{\text {nd}} \:
             \text{moment vector} \: v , \\
            &\: gradients \: g, \: \text{learning rate} \: \gamma,
             \text {exponential decay rates for the moment estimates} \: \beta_{1} \: \beta_{2} , \\
            &\:\text {parameter vector} \: w_{0}, \:\text{timestep} \: t, \: \text{weight decay} \: \lambda \\
            &\textbf{Init}:  m_{0} \leftarrow 0, \: v_{0} \leftarrow 0, \: t \leftarrow 0, \:
             \text{init parameter vector} \: w_{0} \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{repeat} \\
            &\hspace{5mm} t \leftarrow t+1 \\
            &\hspace{5mm}\boldsymbol{g}_{t} \leftarrow \nabla f_{t}\left(\boldsymbol{w}_{t-1}\right) \\
            &\hspace{5mm}\boldsymbol{w}_{t} \leftarrow \boldsymbol{w}_{t-1}-\gamma\lambda\boldsymbol{w}_{t-1} \\
            &\hspace{5mm}\boldsymbol{m}_{t} \leftarrow \beta_{1} \boldsymbol{m}_{t-1}+\left(1-\beta_{1}\right)
             \boldsymbol{g}_{t} \\
            &\hspace{5mm}\boldsymbol{v}_{t} \leftarrow \beta_{2} \boldsymbol{v}_{t-1}+\left(1-\beta_{2}\right)
             \boldsymbol{g}_{t}^{2} \\
            &\hspace{5mm}\widehat{\boldsymbol{m}_{t}} \leftarrow \boldsymbol{m}_{t}/\big(1-\beta_{1}^{t} \big) \\
            &\hspace{5mm}\widehat{\boldsymbol{v}_{t}} \leftarrow \boldsymbol{v}_{t}/\big(1-\beta_{2}^{t} \big) \\
            &\hspace{5mm}\boldsymbol{w}_{t} \leftarrow \boldsymbol{w}_{t-1}-\gamma\widehat{\boldsymbol{m}_{t}}
             /\left(\sqrt{\widehat{\boldsymbol{v}_{t}}}+\epsilon\right) \\
            &\textbf{until}\text { stopping criterion is met } \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
            &\textbf{return} \: \boldsymbol{w}_{t} \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
        \end{array}

    :math:`m` represents the first moment vector moment1, :math:`v` represents the second moment vector moment2,
    :math:`\widehat{m}` represents the bias-corrected first moment vector, :math:`\widehat{v}` represents
    the bias-corrected second moment vector, :math:`g` represents gradients, :math:`\gamma` represents
    learning_rate, :math:`\beta_1`, `\beta_2` represent beta1 and beta2, :math:`t` represents the current step,
    :math:`w` represents params, and :math:`\lambda` represents weight_decay.

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

        use_fused (bool, optional): Whether to enable the fused operator implementation. Default: False.

        amsgrad (bool, optional): Whether to use the AMSGrad variant of the Adam algorithm, which maintains the maximum
            of past squared gradients instead of an exponential moving average. This can help improve model convergence
            in some cases. Only required when use_fused is True; otherwise an error will be raised. If set to True,
            uses the AMSGrad variant of the Adam algorithm. Default: False.

        maximize (bool, optional): Whether to maximize the objective function (rather than minimizing it). This is
            useful for scenarios requiring maximization of reward or utility functions. Only required when use_fused is
            True; otherwise an error will be raised. If set to True, the optimizer will maximize the objective function.
            Default: False.

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

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindformers import AutoModel
        >>> from mindformers.core.optim import AdamW
        >>>
        >>> ms.set_context(mode=ms.context.GRAPH_MODE)
        >>> net = AutoModel.from_pretrained("glm4_9b", num_layers=2)
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = AdamW(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> layernorm_params = list(filter(lambda x: 'norm' in x.name, net.trainable_params()))
        >>> no_layernorm_params = list(filter(lambda x: 'norm' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': layernorm_params, 'weight_decay': 0.01},
        ...                 {'params': no_layernorm_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = AdamW(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The layernorm_params's parameters will use default learning rate of 0.1 and weight decay of 0.01.
        >>> # The no_layernorm_params's parameters will use learning rate of 0.01 and default weight decay of 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim)
    """

    def __init__(
            self,
            params: Union[list[Parameter], list[dict]],
            learning_rate: Optional[Union[float, int, Tensor, Iterable, LearningRateSchedule]],
            betas: Optional[Union[list[float], tuple[float]]],
            eps: Optional[float],
            weight_decay: Optional[Union[float, int, Cell]],
            use_fused: Optional[bool],
            amsgrad: Optional[bool],
            maximize: Optional[bool]
    ):
        pass

    @staticmethod
    def get_actual_adamw_cls(use_fused):
        return FusedAdamW if use_fused else BasicAdamW
