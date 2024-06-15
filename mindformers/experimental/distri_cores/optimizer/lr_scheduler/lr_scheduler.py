# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Learning rate scheduler"""
import numpy as np
import mindspore.common.dtype as mstype

from mindspore import Tensor
from mindspore.nn.learning_rate_schedule import (
    LearningRateSchedule,
    PolynomialDecayLR,
    WarmUpLR,
    CosineDecayLR,
)
from mindspore.ops import operations as P


class LearningRateScheduler(LearningRateSchedule):
    """
    Warmup-decay learning rate for PanguAlpha network.

    Args:
        learning_rate (float): The initial learning rate.
        end_learning_rate (float): The final learning rate after decay.
        warmup_steps (int): The number of warmup steps.
        decay_steps (int): The number of decay steps.
        power (float, optional): The power factor for polynomial decay. Defaults to 1.0.
        use_cosine (bool, optional): Whether to use cosine decay. Defaults to True.

    Inputs:
        - **global_step** (int) - The current global step.

    Examples:
        >>> lr = LearningRateScheduler(0.1, 0.0, 1000, 10000)
        >>> for step in range(0, 11000, 1000):
        ...     print(lr(step))
    """

    def __init__(
            self,
            learning_rate,
            end_learning_rate,
            warmup_steps,
            decay_steps,
            power=1.0,
            use_cosine=True,
        ):
        super(LearningRateScheduler, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr_scheduler = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr_scheduler = (
            PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
            if not use_cosine
            else CosineDecayLR(end_learning_rate, learning_rate, decay_steps)
        )
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """Dynamic learning rate calculation.

        Args:
            global_step (int): The current global step.

        Returns:
            float: The calculated learning rate.
        """
        decay_lr = self.decay_lr_scheduler(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(
                self.greater(self.warmup_steps, global_step), mstype.float32
            )
            warmup_lr = self.warmup_lr_scheduler(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr