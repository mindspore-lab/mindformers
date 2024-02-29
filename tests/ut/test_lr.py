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
# ============================================================================
"""test lr schedule."""

import pytest

import mindspore as ms
from mindspore import Tensor

from mindformers.core.lr import LinearWithWarmUpLR,\
    CosineWithWarmUpLR, PolynomialWithWarmUpLR, \
    CosineWithRestartsAndWarmUpLR, ConstantWarmUpLR, \
    CosineAnnealingLR, CosineAnnealingWarmRestarts


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_lr_schedule():
    """
    Feature: Test LR Schedule
    Description: Test Different LR Schedule
    Expectation: ValueError
    """
    total_steps = 20
    warmup_steps = 10
    learning_rate = 0.005
    lr_end = 0.0000001

    t_max = 10
    t_0 = 10
    t_mult = 2

    constant_with_warmup_lr_std = [0.0005, 0.005, 0.005]
    linear_lr_std = [0.0005, 0.005, 0.0005]
    cosine_lr_std = [0.0005, 0.005, 0.00012236]
    cosine_with_restarts_lr_std = [0.0005, 0.005, 0.00289109]
    polynomial_lr_std = [0.0005, 0.005, 0.00050009]
    cosine_annealing_lr_std = [0.00487764, 0.0000001, 0.00487764]
    cosine_annealing_warm_restarts_lr_std = [0.00487764, 0.005, 0.00289112]

    constant_with_warmup = ConstantWarmUpLR(learning_rate, warmup_steps)
    linear = LinearWithWarmUpLR(learning_rate, warmup_steps, total_steps)
    cosine = CosineWithWarmUpLR(learning_rate, warmup_steps, total_steps)
    cosine_with_restarts = CosineWithRestartsAndWarmUpLR(learning_rate, total_steps, warmup_steps)
    polynomial = PolynomialWithWarmUpLR(learning_rate, warmup_steps, total_steps, lr_end=lr_end)
    cosine_annealing = CosineAnnealingLR(base_lr=learning_rate, t_max=t_max, eta_min=lr_end)
    cosine_annealing_warm_restarts = \
        CosineAnnealingWarmRestarts(base_lr=learning_rate, t_0=t_0, t_mult=t_mult, eta_min=lr_end)

    constant_with_warmup_lr = []
    linear_lr = []
    cosine_lr = []
    cosine_with_restarts_lr = []
    polynomial_lr = []
    cosine_annealing_lr = []
    cosine_annealing_warm_restarts_lr = []

    assert_steps = [1, 10, 19]

    for step in range(0, total_steps):
        if step in assert_steps:
            step = Tensor(step, ms.int32)
            constant_with_warmup_lr.append(constant_with_warmup(step).asnumpy())
            linear_lr.append(linear(step).asnumpy())
            cosine_lr.append(cosine(step).asnumpy())
            cosine_with_restarts_lr.append(cosine_with_restarts(step).asnumpy())
            polynomial_lr.append(polynomial(step).asnumpy())
            cosine_annealing_lr.append(cosine_annealing(step).asnumpy())
            cosine_annealing_warm_restarts_lr.append(cosine_annealing_warm_restarts(step).asnumpy())

    error = 1e-8
    for i in range(3):
        assert abs(constant_with_warmup_lr[i] - constant_with_warmup_lr_std[i]) < error
        assert abs(linear_lr[i] - linear_lr_std[i]) < error
        assert abs(cosine_lr[i] - cosine_lr_std[i]) < error
        assert abs(cosine_with_restarts_lr[i] - cosine_with_restarts_lr_std[i]) < error
        assert abs(polynomial_lr[i] - polynomial_lr_std[i]) < error
        assert abs(cosine_annealing_lr[i] - cosine_annealing_lr_std[i]) < error
        assert abs(cosine_annealing_warm_restarts_lr[i] - cosine_annealing_warm_restarts_lr_std[i]) < error
