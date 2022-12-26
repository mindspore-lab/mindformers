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


class WarmUpCosineDecayV1(LearningRateSchedule):
    def __init__(self, min_lr, max_lr, warmup_steps, decay_steps, start_warmup_value=0.):
        super(WarmUpCosineDecayV1, self).__init__()
        self.schedule = Tensor([lr_adjust(max_lr, min_lr, i, warmup_steps, decay_steps, start_warmup_value)
                                for i in range(warmup_steps + decay_steps)])

    def construct(self, global_step):
        return self.schedule[global_step]
