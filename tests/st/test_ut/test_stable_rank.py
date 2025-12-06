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
    Description: Calc Stable Rank
    Expectation: No Exception
    """
    tensor = ms.ops.randn(200, 200)
    stable_rank, eigenvalue = _get_stable_rank(tensor, 50)
    eigenvalue_ops = ms.ops.square(ms.ops.norm(tensor, ord=2))
    f_norm = ms.ops.norm(tensor, ord='fro', dim=(-2, -1))
    stable_rank_ops = ms.ops.square(f_norm).asnumpy() / eigenvalue_ops
    assert abs(stable_rank - stable_rank_ops) < (stable_rank * 0.05)
    assert abs(eigenvalue - eigenvalue_ops) < (eigenvalue * 0.05)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_3d_param_stable_rank():
    """
    Feature: 3d Rand Param
    Description: Calc Stable Rank when 3d Rand Param
    Expectation: No Exception
    """
    tensor = ms.ops.randn((6, 30, 40), dtype=ms.float32)
    tensor[1] = 0.0
    tensor[3] = 0.0
    tensor[5] = 0.0
    sr, eig = _get_stable_rank(tensor, 50)

    tensor0 = tensor[0]
    sr0, eig0 = _get_stable_rank(tensor0, 50)

    tensor2 = tensor[2]
    sr2, eig2 = _get_stable_rank(tensor2, 50)

    tensor4 = tensor[4]
    sr4, eig4 = _get_stable_rank(tensor4, 50)

    assert abs(sr[0] - sr0) < (sr0 * 0.05)
    assert abs(eig[0] - eig0) < (eig0 * 0.05)
    assert sr[1] == 0.0
    assert eig[1] == 0.0
    assert abs(sr[2] - sr2) < (sr0 * 0.05)
    assert abs(eig[2] - eig2) < (eig0 * 0.05)
    assert sr[3] == 0.0
    assert eig[3] == 0.0
    assert abs(sr[4] - sr4) < (sr0 * 0.05)
    assert abs(eig[4] - eig4) < (eig0 * 0.05)
    assert sr[5] == 0.0
    assert eig[5] == 0.0

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_2d_zero_stable_rank():
    """
    Feature: 2d Zero Param
    Description: Calc Stable Rank when 2d Zero Param
    Expectation: No Exception
    """
    tensor = ms.ops.zeros((30, 40), dtype=ms.float32)
    sr, eig = _get_stable_rank(tensor, 50)
    assert (sr, eig) == (0.0, 0.0)
