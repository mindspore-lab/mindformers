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
# ============================================================================
"""Learning rate utilities."""

import math
import numpy as np
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR


def linear_warmup(warmup_steps, current_step):
    return min([1.0, float(current_step)/float(warmup_steps)])


def rsqrt_decay(warmup_steps, current_step):
    return float(max([current_step, warmup_steps])) ** -0.5


def rsqrt_hidden(hidden_size):
    return float(hidden_size) ** -0.5


def create_dynamic_lr(schedule, training_steps, learning_rate, warmup_steps, hidden_size,
                      start_decay_step=0, min_lr=0.):
    """
    Generate dynamic learning rate.
    """
    if start_decay_step < warmup_steps:
        start_decay_step = warmup_steps
    lr = []
    for current_step in range(1, training_steps+1):
        cur_lr = 1.0
        for name in schedule.split("*"):
            if name == "constant":
                cur_lr *= float(learning_rate)
            elif name == "rsqrt_hidden":
                cur_lr *= rsqrt_hidden(hidden_size)
            elif name == "linear_warmup":
                cur_lr *= linear_warmup(warmup_steps, current_step)
            elif name == "rsqrt_decay":
                cur_lr *= rsqrt_decay(warmup_steps, current_step-start_decay_step+warmup_steps)
            else:
                raise ValueError("unknown learning rate schedule")
        if warmup_steps < current_step < start_decay_step:
            cur_lr = lr[-1]
        if current_step > warmup_steps:
            cur_lr = max([cur_lr, min_lr])
        lr.append(cur_lr)
    return lr


class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for GPT network.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power=1.0, use_cosine=True):
        super(LearningRate, self).__init__()
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


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, \
           total_epochs, steps_per_epoch, lr_decay_mode, poly_power=2.0):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(int): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch
       lr_decay_mode(string): learning rate decay mode, including steps, poly, cosine or default

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = int(steps_per_epoch * warmup_epochs)
    if lr_decay_mode == 'steps':
        decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps]
        for i in range(total_steps):
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * 0.1
            elif i < decay_epoch_index[2]:
                lr = lr_max * 0.01
            else:
                lr = lr_max * 0.001
            lr_each_step.append(lr)
    elif lr_decay_mode == 'poly':
        if warmup_steps != 0:
            inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
        else:
            inc_each_step = 0
        for i in range(total_steps):
            if i < warmup_steps:
                lr = float(lr_init) + inc_each_step * float(i)
            else:
                base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
                lr = float(lr_max - lr_end) * base ** poly_power + lr_end
                lr = max(lr, 0.0)
            lr_each_step.append(lr)
    elif lr_decay_mode == 'cosine':
        decay_steps = total_steps - warmup_steps
        for i in range(total_steps):
            if i < warmup_steps:
                lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
                lr = float(lr_init) + lr_inc * (i + 1)
            else:
                cur_step = i + 1 - warmup_steps
                lr = lr_max * (1 + math.cos(math.pi * cur_step / decay_steps)) / 2
            lr_each_step.append(lr)
    else:
        for i in range(total_steps):
            if i < warmup_steps:
                lr = lr_init + (lr_max - lr_init) * i / warmup_steps
            else:
                lr = lr_max - (lr_max - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
            lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def build_lr(config, epoch_num, step_per_epoch):
    """Build the learning rate according to the input arguments"""
    model_name = config.arch
    lr = None
    if model_name in ['bert', 'gpt', 'opt']:
        lr = LearningRate(learning_rate=float(config.start_lr),
                          end_learning_rate=float(config.end_lr),
                          warmup_steps=config.warmup_step,
                          decay_steps=epoch_num * step_per_epoch)
    elif model_name in ['t5']:
        learning_rate = config.learning_rate if config.context['device_target'] == "Ascend" else 1.0
        lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                      training_steps=step_per_epoch*epoch_num,
                                      learning_rate=learning_rate,
                                      warmup_steps=config.warmup_steps,
                                      hidden_size=config.model['hidden_size'],
                                      start_decay_step=config.start_decay_step,
                                      min_lr=config.min_lr), mstype.float32)
    else:
        raise RuntimeError(f"Model name {model_name} is not supported yet.")

    return lr
