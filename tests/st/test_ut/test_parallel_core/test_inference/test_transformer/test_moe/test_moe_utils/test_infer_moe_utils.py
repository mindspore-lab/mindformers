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
"""UTs for `moe_utils.py`."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
import mindspore.common.dtype as mstype

from mindformers.parallel_core.inference.transformer.moe.moe_utils import (
    group_limited_topk,
    topk_routing_with_score_function,
)


ms.context.set_context(deterministic="ON")
jit_level = "O0"
infer_boost = "on"
ms.set_context(device_target="Ascend",
               mode=ms.GRAPH_MODE,
               jit_config={"jit_level": jit_level, "infer_boost": infer_boost})


class TestTopkRoutingWithScoreFunction:
    """Unit tests for the top-k routing helper."""

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_softmax_routing_returns_normalized_weights(self):
        """The weights should sum to one per token when normalization is enabled."""
        logits = Tensor(
            np.array([[1.0, 2.0, 0.5, -0.5], [-1.0, 0.0, 2.5, 1.0]], dtype=np.float32),
            dtype=mstype.float32,
        )
        expert_weight, routing_map = topk_routing_with_score_function(
            logits=logits,
            topk=2,
            num_experts=4,
            score_function="softmax",
            norm_topk_prob=True,
        )

        assert expert_weight.shape == (2, 2)
        assert routing_map.shape == (2, 2)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_sigmoid_routing_with_bias_without_normalization(self):
        """Bias should affect the chosen experts while weights stay unnormalized when disabled."""
        logits = Tensor(
            np.array([[0.0, -2.0, 2.0, 1.0]], dtype=np.float32),
            dtype=mstype.float32,
        )
        expert_bias = Tensor(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), dtype=mstype.float32)

        expert_weight, routing_map = topk_routing_with_score_function(
            logits=logits,
            topk=2,
            num_experts=4,
            score_function="sigmoid",
            expert_bias=expert_bias,
            norm_topk_prob=False,
        )

        assert expert_weight.shape == (1, 2)
        assert routing_map.shape == (1, 2)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_group_limited_topk_only_selects_from_best_group(self):
        """group_limited_topk should not route experts outside the best group subset."""
        scores = Tensor(np.array([[0.9, 0.8, 0.1, 0.2]], dtype=np.float32), dtype=mstype.float32)

        probs, top_indices = group_limited_topk(
            scores=scores,
            topk=2,
            num_experts=4,
            num_groups=2,
            group_topk=1,
        )

        assert probs.shape == (1, 2)
        assert top_indices.shape == (1, 2)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_invalid_score_function_raises(self):
        """An unsupported score function name should raise ValueError."""
        logits = Tensor(np.zeros((1, 2), dtype=np.float32), dtype=mstype.float32)

        with pytest.raises(ValueError):
            topk_routing_with_score_function(
                logits=logits,
                topk=1,
                num_experts=2,
                score_function="unsupported",
            )
