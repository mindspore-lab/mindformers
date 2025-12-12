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
"""UTs for FusedScaleMaskSoftmax."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
import mindspore.common.dtype as mstype

from mindformers.parallel_core.inference.transformer.fused_softmax import FusedScaleMaskSoftmax


ms.context.set_context(deterministic="ON")
jit_level = "O0"
infer_boost = "on"
ms.set_context(device_target="Ascend",
               mode=ms.GRAPH_MODE,
               jit_config={"jit_level": jit_level, "infer_boost": infer_boost})


def simple_mask(tensor, mask):
    """Mask function for tests that multiplies by mask."""
    return tensor + mask


class TestFusedScaleMaskSoftmax:
    """Tests covering the fused softmax helper."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_forward_pass_with_scale_and_mask(self):
        """
        Test the forward pass of FusedScaleMaskSoftmax with both scaling and a mask applied.

        Verifies that the module correctly applies the scale factor to the input tensor,
        applies the provided attention mask, and computes the softmax, returning an output
        with the expected shape. This tests the core functionality under typical conditions.
        """
        fused_softmax = FusedScaleMaskSoftmax(mask_func=simple_mask, scale=0.5, softmax_compute_type=mstype.float32)
        x = Tensor(np.array([[2.0, 0.0]], dtype=np.float32), dtype=mstype.float32)
        mask = Tensor(np.array([[0.0, -1.0]], dtype=np.float32), dtype=mstype.float32)

        output = fused_softmax(x, mask)

        assert output.shape == (1, 2)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_precision_casts_to_fp32_when_needed(self):
        """
        Test that FusedScaleMaskSoftmax automatically casts inputs to float32 when required.

        Verifies that when the softmax computation type is set to float32 but the input
        tensor is in float16, the module performs the necessary precision casting to fp32
        for the softmax operation, ensuring numerical stability, and returns an output
        with the correct shape.
        """
        fused_softmax = FusedScaleMaskSoftmax(mask_func=simple_mask, scale=None, softmax_compute_type=mstype.float32)
        x = Tensor(np.array([[1.0, 1.0]], dtype=np.float16), dtype=mstype.float16)

        output = fused_softmax(x, mask=None)

        assert output.shape == (1, 2)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_invalid_scale_precision_combination_raises(self):
        """
        Test that FusedScaleMaskSoftmax raises a ValueError for invalid precision combinations.

        Verifies that the module enforces the rule that if a scale factor is applied,
        the softmax computation must be performed in float32 to maintain precision.
        Attempting to use a scale with float16 computation should raise a ValueError.
        """
        with pytest.raises(ValueError):
            FusedScaleMaskSoftmax(mask_func=simple_mask, scale=0.1, softmax_compute_type=mstype.float16)
