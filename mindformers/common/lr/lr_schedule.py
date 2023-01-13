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
# This file was refer to project:
# https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/PanGu-%CE%B1/utils.py
# ============================================================================
"""Self-Define LR Schedule."""
import math

import numpy as np
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.common.tensor import Tensor
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['LearningRateWiseLayer', 'WarmUpDecayLR', 'WarmUpCosineDecayV1']


@MindFormerRegister.register(MindFormerModuleType.LR)
class LearningRateWiseLayer(LearningRateSchedule):
    """LearningRateWiseLayer."""

    def __init__(self, base_lr, lr_scale):
        super(LearningRateWiseLayer, self).__init__()
        self.base_lr = base_lr
        self.lr_scale = lr_scale

    def construct(self, global_step):
        lr = self.base_lr(global_step)
        return self.lr_scale * lr

@MindFormerRegister.register(MindFormerModuleType.LR)
class WarmUpDecayLR(LearningRateSchedule):
    """
    Warmup-decay learning rate for Bert network.
    """

    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power=1.0, use_cosine=False):
        super(WarmUpDecayLR, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate, decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


def lr_adjust(max_lr, min_lr, step, warmup_steps, decay_steps, start_warmup_value=0.):
    if step < warmup_steps:
        lr = max_lr * step / warmup_steps + start_warmup_value
    else:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * (step - warmup_steps) / decay_steps))
    return lr


@MindFormerRegister.register(MindFormerModuleType.LR)
class WarmUpCosineDecayV1(LearningRateSchedule):
    """WarmUpCosineDecayV1 learning rate"""
    def __init__(self, min_lr, base_lr, warmup_steps, decay_steps, warmup_lr=0.):
        super(WarmUpCosineDecayV1, self).__init__()
        self.schedule = Tensor([lr_adjust(base_lr, min_lr, i, warmup_steps, decay_steps, warmup_lr)
                                for i in range(warmup_steps + decay_steps)])

    def construct(self, global_step):
        return self.schedule[global_step]


@MindFormerRegister.register(MindFormerModuleType.LR)
class WarmUpCosineDecayV2(LearningRateSchedule):
    # This class was refer to project:
    # https://github.com/rwightman/pytorch-image-models/blob/main/timm/scheduler/cosine_lr.py
    """
    WarmUpCosineDecayV2 learning rate
    """
    def __init__(self,
                 base_lr: float,
                 total_steps: int,
                 min_lr: float = 0.,
                 cycle_mul: float = 1.,
                 cycle_decay: float = 1.,
                 cycle_limit: int = 1,
                 warmup_steps=0,
                 warmup_lr=0.,
                 warmup_prefix=False,
                 t_in_epochs=True,
                 k_decay=1.0) -> None:
        super(WarmUpCosineDecayV2, self).__init__()

        assert total_steps > 0
        assert min_lr >= 0

        self.t_initial = total_steps
        self.min_lr = min_lr
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        self.k_decay = k_decay

        self.base_values = [base_lr,]
        if self.warmup_steps:
            self.warmup_t = [(v - warmup_lr) / self.warmup_steps for v in self.base_values]
        else:
            self.warmup_t = [1 for _ in self.base_values]

        self.lr_tensor = Tensor([self.get_epoch_values(i) for i in range(self.t_initial)], mstype.float32)

    def _get_lr(self, t):
        """get lr"""
        if t < self.warmup_steps:
            lrs = [self.warmup_lr + t * s for s in self.warmup_t]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_steps

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
                    self.min_lr + 0.5 * (lr_max - self.min_lr) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.min_lr for _ in self.base_values]

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
