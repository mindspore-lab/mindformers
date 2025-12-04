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
"""UTs for tensor-parallel mapping helpers."""
from functools import partial

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
import mindspore.common.dtype as mstype

from mindformers.parallel_core.inference.parallel_state import ProcessGroup
from mindformers.parallel_core.inference.tensor_parallel import mappings


ms.context.set_context(deterministic="ON")
jit_level = "O0"
infer_boost = "on"
ms.set_context(device_target="Ascend",
               mode=ms.GRAPH_MODE,
               jit_config={"jit_level": jit_level, "infer_boost": infer_boost})


class FakeGather:
    """Mock AllGather operator recording inputs."""

    def __init__(self):
        self.calls = []

    def __call__(self, tensor):
        self.calls.append(tensor)
        return tensor


class FakeReduceScatter:
    """Mock ReduceScatter returning half-size tensor."""

    def __init__(self):
        self.calls = []

    def __call__(self, tensor):
        self.calls.append(tensor)
        # Return the first split chunk
        return tensor[:tensor.shape[0] // 2]


class FakeAllReduce:
    """Mock AllReduce returning tensor doubled."""

    def __init__(self):
        self.calls = []

    def __call__(self, tensor):
        self.calls.append(tensor)
        return tensor * 2


class FakeSplit:
    """Mock Split op returning chunks."""

    def __init__(self, axis, output_num):
        self.axis = axis
        self.output_num = output_num

    def __call__(self, tensor):
        return tuple(np.split(tensor.asnumpy(), self.output_num, axis=self.axis))

class TestTensorParallelMappings:
    """Groups mapping tests into a single suite."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_gather_returns_input_when_group_size_one(self):
        """
        Test that gather_from_model_parallel_region returns the original tensor unchanged
        when the process group size is 1.
        """
        group = ProcessGroup(group=None, rank=0, size=1)
        # pylint: disable=W0212
        group._is_group_created = True
        tensor = Tensor(np.ones((2, 2), dtype=np.float32), dtype=mstype.float32)

        output = mappings.gather_from_model_parallel_region(tensor, group, dim=-1)

        assert np.array_equal(output.asnumpy(), tensor.asnumpy())


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_gather_transposes_when_dim_nonzero(self, monkeypatch):
        """
        Test that gather_from_model_parallel_region correctly handles gathering along a non-last dimension.
        """
        fake_gather = FakeGather()
        monkeypatch.setattr(mappings.ops, "AllGather", lambda group: fake_gather)
        group = ProcessGroup(group="test", rank=0, size=2)
        # pylint: disable=W0212
        group._is_group_created = True
        tensor = Tensor(np.arange(6).reshape(3, 2).astype(np.float32), dtype=mstype.float32)

        output = mappings.gather_from_model_parallel_region(tensor, group, dim=1)

        assert output.shape == tensor.shape


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_reduce_allreduce_invoked(self, monkeypatch):
        """
        Test that reduce_from_model_parallel_region performs an AllReduce operation.
        """
        fake_reduce = FakeAllReduce()
        monkeypatch.setattr(mappings.ops, "AllReduce", lambda group: fake_reduce)
        group = ProcessGroup(group="test", rank=0, size=2)
        # pylint: disable=W0212
        group._is_group_created = True
        tensor = Tensor(np.ones((2, 2), dtype=np.float32), dtype=mstype.float32)

        output = mappings.reduce_from_model_parallel_region(tensor, group)

        assert np.array_equal(output.asnumpy(), (tensor * 2).asnumpy())


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_reduce_scatter_returns_split(self, monkeypatch):
        """
        Test that reduce_scatter_to_model_parallel_region performs a ReduceScatter operation.
        """
        fake_reduce_scatter = FakeReduceScatter()
        monkeypatch.setattr(mappings.ops, "ReduceScatter", lambda group: fake_reduce_scatter)
        group = ProcessGroup(group="test", rank=0, size=2)
        # pylint: disable=W0212
        group._is_group_created = True
        tensor = Tensor(np.ones((4, 2), dtype=np.float32), dtype=mstype.float32)

        output = mappings.reduce_scatter_to_model_parallel_region(tensor, group)

        assert output.shape[0] == tensor.shape[0] // 2


    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_scatter_returns_rank_chunk(self, monkeypatch):
        """
        Test that scatter_to_model_parallel_region splits the input tensor along the specified dimension.
        """
        monkeypatch.setattr(mappings.ops, "Split", partial(FakeSplit))
        group = ProcessGroup(group="test", rank=1, size=2)
        # pylint: disable=W0212
        group._is_group_created = True
        tensor = Tensor(np.arange(8).reshape(2, 4).astype(np.float32), dtype=mstype.float32)

        output = mappings.scatter_to_model_parallel_region(tensor, group, dim=1)

        assert output.shape == (2, 2)
