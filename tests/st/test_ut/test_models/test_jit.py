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
"""test jit."""
import os
import pytest
import numpy as np

import mindspore as ms
from mindspore.nn import Cell
from mindformers.tools.utils import is_pynative
from mindformers.models.utils import jit


class JitCell(Cell):
    """ jit cell """

    @jit
    def construct(self, x, y):
        return x * y


class TestJit:
    """A test class for jit."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_jit_graph_mode(self):
        """test jit for graph mode."""
        os.environ["RUN_MODE"] = "predict"
        cell = JitCell()
        x = np.random.rand(10, 10).astype(np.float32)
        y = np.random.rand(10, 10).astype(np.float32)
        res = cell(ms.Tensor(x).astype(ms.float32), ms.Tensor(y).astype(ms.float32))
        assert (res.asnumpy() == x * y).all()

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_jit_pynative_mode(self):
        """test jit for pynative mode."""
        os.environ["RUN_MODE"] = "predict"
        os.environ['FORCE_EAGER'] = "True"
        is_pynative_mode = is_pynative()
        assert is_pynative_mode

        cell = JitCell()
        x = np.random.rand(10, 10).astype(np.float32)
        y = np.random.rand(10, 10).astype(np.float32)
        res = cell(ms.Tensor(x).astype(ms.float32), ms.Tensor(y).astype(ms.float32))
        assert (res.asnumpy() == x * y).all()
