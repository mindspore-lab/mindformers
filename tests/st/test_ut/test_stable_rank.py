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
"""test stable rank calculation."""
import pytest

import mindspore as ms
from mindformers.core.callback.callback import _get_stable_rank

ms.set_context(mode=1, device_target='Ascend')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_stable_rank():
    """
    Feature: Stable Rank
    Description: Max Stable Rank
    Expectation: No Exception
    """
    tensor = ms.ops.randn(200, 200)
    stable_rank, eigenvalue = _get_stable_rank(tensor, 50)
    eigenvalue_ops = ms.ops.square(ms.ops.norm(tensor, ord=2))
    f_norm = ms.ops.norm(tensor, ord='fro', dim=(-2, -1))
    stable_rank_ops = ms.ops.square(f_norm).asnumpy() / eigenvalue_ops
    assert abs(stable_rank - stable_rank_ops) < (stable_rank * 0.05)
    assert abs(eigenvalue - eigenvalue_ops) < (eigenvalue * 0.05)
