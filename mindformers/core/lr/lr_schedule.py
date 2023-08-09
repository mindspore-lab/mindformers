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
# This file was refer to project:
# https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/PanGu-%CE%B1/utils.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py
# ============================================================================
"""Self-Define LR Schedule."""
import math

import numpy as np
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.common.tensor import Tensor
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = [
    'LearningRateWiseLayer', 'WarmUpDecayLR', 'ConstantWarmUpLR',
    'LinearWithWarmUpLR', 'CosineWithWarmUpLR',
    'CosineWithRestartsAndWarmUpLR', 'PolynomialWithWarmUpLR']


@MindFormerRegister.register(MindFormerModuleType.LR)
class ConstantWarmUpLR(LearningRateSchedule):
    """ConstantWarmUpLR."""

    def __init__(self, learning_rate: float, warmup_steps: int, warmup_lr_init: float = 0., **kwargs):
        super(ConstantWarmUpLR, self).__init__()
        warmup_steps = max(1, warmup_steps)
        self.learning_rate = learning_rate
        self.warmup_lr_init = warmup_lr_init
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.one_constant = Tensor(1.0, mstype.float32)
        self.greater = P.Greater()
        self.total_steps = kwargs.get('total_steps', None)

    def construct(self, global_step):
        """compute current step lr."""
        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + self.learning_rate * percent
        else:
            percent = self.one_constant
            learning_rate = self.learning_rate * percent
        return learning_rate


@MindFormerRegister.register(MindFormerModuleType.LR)
class LinearWithWarmUpLR(LearningRateSchedule):
    """LinearWithWarmUpLR."""

    def __init__(self, learning_rate: float, warmup_steps: int, total_steps: int,
                 warmup_lr_init: float = 0.):
        super(LinearWithWarmUpLR, self).__init__()
        linear_steps = max(1, total_steps - warmup_steps)
        warmup_steps = max(1, warmup_steps)
        self.learning_rate = learning_rate
        self.warmup_lr_init = warmup_lr_init
        self.total_steps = Tensor(total_steps, mstype.float32)
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.linear_steps = Tensor(linear_steps, mstype.float32)
        self.greater = P.Greater()
        self.max = P.Maximum()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + self.learning_rate * percent
        else:
            percent = self.max(self.zero_constant, (self.total_steps - global_step) / self.linear_steps)
            learning_rate = self.learning_rate * percent
        return learning_rate


@MindFormerRegister.register(MindFormerModuleType.LR)
class CosineWithWarmUpLR(LearningRateSchedule):
    """CosineWithWarmUpLR."""

    def __init__(self, learning_rate: float, warmup_steps: int, total_steps: int,
                 num_cycles: float = 0.5, lr_end: float = 0., warmup_lr_init: float = 0.):
        super(CosineWithWarmUpLR, self).__init__()
        cosine_steps = max(1, total_steps - warmup_steps)
        warmup_steps = max(1, warmup_steps)
        self.learning_rate = learning_rate
        self.lr_end = lr_end
        self.warmup_lr_init = warmup_lr_init
        self.total_steps = Tensor(total_steps, mstype.float32)
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.cosine_steps = Tensor(cosine_steps, mstype.float32)
        self.num_cycles = num_cycles
        self.greater = P.Greater()
        self.max = P.Maximum()
        self.math_pi = math.pi
        self.cos = P.Cos()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + self.learning_rate * percent
        else:
            progress = (global_step - self.warmup_steps) / self.cosine_steps
            percent = self.max(
                self.zero_constant, 0.5 * (1.0 + self.cos(self.math_pi * self.num_cycles * 2.0 * progress)))
            learning_rate = self.lr_end + (self.learning_rate - self.lr_end) * percent
        return learning_rate


@MindFormerRegister.register(MindFormerModuleType.LR)
class CosineWithRestartsAndWarmUpLR(LearningRateSchedule):
    """CosineWithRestartsAndWarmUpLR."""

    def __init__(self, learning_rate: float, total_steps: int, warmup_steps: int,
                 num_cycles: float = 0.5, lr_end: float = 0., warmup_lr_init: float = 0.):
        super(CosineWithRestartsAndWarmUpLR, self).__init__()
        cosine_steps = max(1, total_steps - warmup_steps)
        warmup_steps = max(1, warmup_steps)
        self.learning_rate = learning_rate
        self.lr_end = lr_end
        self.warmup_lr_init = warmup_lr_init
        self.total_steps = Tensor(total_steps, mstype.float32)
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.cosine_steps = Tensor(cosine_steps, mstype.float32)
        self.num_cycles = num_cycles
        self.greater = P.Greater()
        self.max = P.Maximum()
        self.math_pi = math.pi
        self.cos = P.Cos()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.one_constant = Tensor(1.0, mstype.float32)
        self.cast = P.Cast()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + self.learning_rate * percent
        else:
            progress = (global_step - self.warmup_steps) / self.cosine_steps
            if self.greater(self.one_constant, progress):
                percent = self.max(
                    self.zero_constant,
                    0.5 * (1.0 + self.cos(self.math_pi * ((self.num_cycles * progress) % self.one_constant))))
                learning_rate = self.lr_end + (self.learning_rate - self.lr_end) * percent
            else:
                return self.zero_constant
        return learning_rate


@MindFormerRegister.register(MindFormerModuleType.LR)
class PolynomialWithWarmUpLR(LearningRateSchedule):
    """PolynomialWithWarmUpLR."""

    def __init__(self, learning_rate: float, warmup_steps: int, total_steps: int,
                 lr_end: float = 1e-7, power: float = 1.0, warmup_lr_init: float = 0.):
        super(PolynomialWithWarmUpLR, self).__init__()
        decay_steps = max(1, total_steps - warmup_steps)
        warmup_steps = max(1, warmup_steps)
        if not learning_rate > lr_end:
            raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({learning_rate})")
        self.learning_rate = learning_rate
        self.warmup_lr_init = warmup_lr_init
        self.lr_end = lr_end
        self.power = power
        self.total_steps = Tensor(total_steps, mstype.float32)
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.decay_steps = Tensor(decay_steps, mstype.float32)
        self.greater = P.Greater()
        self.cast = P.Cast()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + self.learning_rate * percent
        else:
            lr_range = self.learning_rate - self.lr_end
            pct_remaining = 1 - (global_step - self.warmup_steps) / self.decay_steps
            decay = lr_range * pct_remaining ** self.power + self.lr_end
            percent = decay / self.learning_rate
            learning_rate = self.learning_rate * percent
        return learning_rate


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


@MindFormerRegister.register(MindFormerModuleType.LR)
class NoamLR(LearningRateSchedule):
    def __init__(self, learning_rate, warmup_iter, end_iter, total_steps=-1):
        super(NoamLR, self).__init__()
        self.learning_rate = learning_rate
        self.warmup_iter = warmup_iter
        self.end_iter = end_iter
        self.total_steps = total_steps
        self.scalar_to_tensor = P.ScalarToTensor()
        self.sqrt = P.Sqrt()

    def get_lr_warmup(self, num_iter):
        return self.learning_rate / self.sqrt(
            self.scalar_to_tensor(self.warmup_iter, mstype.float32)) * num_iter / self.warmup_iter

    def get_lr_decay(self, num_iter):
        return self.learning_rate / self.sqrt(num_iter.to(mstype.float32))

    def construct(self, global_step):
        if global_step < self.warmup_iter:
            return self.get_lr_warmup(global_step)
        else:
            return self.get_lr_decay(global_step)
