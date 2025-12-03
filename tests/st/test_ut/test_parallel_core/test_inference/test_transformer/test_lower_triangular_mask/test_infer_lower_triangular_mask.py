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
"""Unit tests for LowerTriangularMaskWithDynamic."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
import mindspore.common.dtype as mstype

from mindformers.parallel_core.inference.transformer.lower_triangular_mask import (
    LowerTriangularMaskWithDynamic,
)


ms.context.set_context(deterministic="ON")
jit_level = "O0"
infer_boost = "on"
ms.set_context(device_target="Ascend",
               mode=ms.GRAPH_MODE,
               jit_config={"jit_level": jit_level, "infer_boost": infer_boost})


class TestLowerTriangularMask:
    """Validates lower-triangular mask generation."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_lower_triangular_mask_in_prefill(self):
        """Prefill path should directly return the static fa mask."""
        lower_triangular_mask = LowerTriangularMaskWithDynamic(seq_length=4, compute_type=mstype.float16)
        lower_triangular_mask.is_prefill = True

        mask = lower_triangular_mask(positions=Tensor(np.zeros((1, 4)), dtype=mstype.int32))
        assert mask.shape == lower_triangular_mask.fa_lower_triangle_mask.shape

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_lower_triangular_mask_in_decode(self):
        """Decode path should gather using provided positions."""
        lower_triangular_mask = LowerTriangularMaskWithDynamic(seq_length=4, compute_type=mstype.float16)
        lower_triangular_mask.is_prefill = False
        positions = Tensor(np.array([0, 2], dtype=np.int32))

        mask = lower_triangular_mask(positions=positions)
        expected_shape = (positions.shape[0], lower_triangular_mask.pa_lower_triangle_mask.shape[1])
        assert mask.shape == expected_shape

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_lower_triangular_mask_bfloat16_in_prefill_mask(self):
        """When using bf16 compute type, mask coefficient becomes +1."""
        lower_triangular_mask = LowerTriangularMaskWithDynamic(seq_length=4, compute_type=mstype.bfloat16)
        mask = lower_triangular_mask.prefill()
        assert mask.shape == lower_triangular_mask.fa_lower_triangle_mask.shape
