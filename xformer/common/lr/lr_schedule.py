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
"""lr schedule"""
import math

import mindspore.common.dtype as mstype
import numpy as np
from mindspore import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, WarmUpLR
from mindspore.ops import operations as P

from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.LR)
class LearningRateWiseLayer(LearningRateSchedule):
    def __init__(self, base_lr, lr_scale):
        super(LearningRateWiseLayer, self).__init__()
        self.base_lr = base_lr
        self.lr_scale = lr_scale

    def construct(self, global_step):
        lr = self.base_lr(global_step)
        return self.lr_scale * lr


@XFormerRegister.register(XFormerModuleType.LR)
class WarmUpCosineDecayV1(LearningRateSchedule):
    def __init__(self, min_lr, max_lr, warmup_steps, decay_steps, start_warmup_lr=0.):
        super(WarmUpCosineDecayV1, self).__init__()
        self.schedule = Tensor([lr_adjust(max_lr, min_lr, i, warmup_steps, decay_steps, start_warmup_lr)
                                for i in range(warmup_steps + decay_steps)])

    def construct(self, global_step):
        return self.schedule[global_step]


@XFormerRegister.register(XFormerModuleType.LR)
class MultiEpochsDecayLR(LearningRateSchedule):  # for simmim vit.
    """MultiEpochsDecayLR"""

    def __init__(self, learning_rate, multi_epochs, steps_per_epoch=1, factor=10):
        super(MultiEpochsDecayLR, self).__init__()
        if not isinstance(multi_epochs, (list, tuple)):
            raise TypeError("multi_epochs must be list or tuple.")
        self.multi_epochs = Tensor(np.array(multi_epochs, dtype=np.float32) * steps_per_epoch)
        self.num = len(multi_epochs)
        self.start_learning_rate = learning_rate
        self.factor = factor
        self.pow = P.Pow()
        self.cast = P.Cast()
        self.less_equal = P.LessEqual()
        self.reduce_sum = P.ReduceSum()

    def construct(self, global_step):
        cur_step = self.cast(global_step, mstype.float32)
        epochs = self.cast(self.less_equal(self.multi_epochs, cur_step), mstype.float32)
        lr = self.start_learning_rate / self.pow(self.factor, self.reduce_sum(epochs, ()))
        return lr


@XFormerRegister.register(XFormerModuleType.LR)
class WarmUpMultiStepDecay(LearningRateSchedule):
    """WarmUpMultiStepDecay"""

    def __init__(self, base_lr, warmup_steps, start_warmup_value,
                 factor=10, multi_epochs=None, steps_per_epoch=1):
        super(WarmUpMultiStepDecay, self).__init__()
        if multi_epochs is None:
            multi_epochs = [700, ]
        self.warmup_lr = WarmUpLR(base_lr, warmup_steps + 1, start_warmup_value)
        self.multisteps_lr = MultiEpochsDecayLR(base_lr, multi_epochs, steps_per_epoch, factor)

        self.warmup_steps = warmup_steps

    def construct(self, global_step):
        if global_step < self.warmup_steps:
            lr = self.warmup_lr(global_step)
        else:
            lr = self.multisteps_lr(global_step)

        return lr


@XFormerRegister.register(XFormerModuleType.LR)
class WarmUpLRV2(LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, start_warmup_value=0.):
        super(WarmUpLRV2, self).__init__()
        self.warmup_schedule = Tensor(
            np.linspace(start_warmup_value, base_lr, warmup_steps), mstype.float32)

    def construct(self, global_step):
        return self.warmup_schedule[global_step]


@XFormerRegister.register(XFormerModuleType.LR)
class CosineDecayLRV2(LearningRateSchedule):
    def __init__(self, min_lr, max_lr, decay_steps):
        super(CosineDecayLRV2, self).__init__()
        self.cosine_schedule = Tensor(
            [min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * i / decay_steps))
             for i in range(decay_steps)], mstype.float32)

    def construct(self, global_step):
        return self.cosine_schedule[global_step]


@XFormerRegister.register(XFormerModuleType.LR)
class WarmUpCosineDecayV2(LearningRateSchedule):
    """WarmUpCosineDecayV2"""

    def __init__(self,
                 base_lr: float,
                 t_initial: int,
                 lr_min: float = 0.,
                 cycle_mul: float = 1.,
                 cycle_decay: float = 1.,
                 cycle_limit: int = 1,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 warmup_prefix=False,
                 t_in_epochs=True,
                 k_decay=1.0) -> None:
        super(WarmUpCosineDecayV2, self).__init__()

        assert t_initial > 0
        assert lr_min >= 0

        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        self.k_decay = k_decay

        self.base_values = [base_lr, ]
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            # super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

        self.lr_tensor = Tensor([self.get_epoch_values(i) for i in range(self.t_initial)], mstype.float32)

    def _get_lr(self, t):
        """get lr"""
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i1 = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i1 * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i1) / (1 - self.cycle_mul) * self.t_initial
            else:
                i1 = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i1)

            gamma = self.cycle_decay ** i1
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i1 < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        return None

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))

    def construct(self, global_step):
        return self.lr_tensor[global_step][0]


def lr_adjust(max_lr, min_lr, step, warmup_steps, decay_steps, start_warmup_value=0.):
    if step < warmup_steps:
        lr = max_lr * step / warmup_steps + start_warmup_value
    else:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (
                1. + math.cos(math.pi * (step - warmup_steps) / decay_steps))
    return lr
