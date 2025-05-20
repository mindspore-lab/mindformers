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
"""Test module for testing dropout for mindformers."""
import pytest
import mindspore as ms
from mindformers.parallel_core.training_graph.transformer.dropout import Dropout


class TestDropout:
    """A test class for testing mcore Dropout"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.input = ms.ops.randn((2, 4, 16), dtype=ms.float32)
        self.dropout = Dropout()
        self.dropout_0 = Dropout(drop_prob=0.0)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_p_0_5_case(self):
        """
        Feature: Dropout
        Description: Test Base Case: drop_prob=0.5,  training=True
        Exception: AssertionError
        """
        self.dropout.set_train(True)
        output = self.dropout(self.input)
        assert not ms.ops.equal(output, self.input).all(), (
            f"Dropout output should not be equal to input when training=True and drop_prob=0.5.\n"
            f"Input:\n{self.input}\n"
            f"Output:\n{output}"
        )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_training_false_case(self):
        """
        Feature: Dropout
        Description: Test Base Case: drop_prob=0.5,  training=False
        Exception: AssertionError
        """
        self.dropout.set_train(False)
        output = self.dropout(self.input)
        assert ms.ops.equal(output, self.input).all(), (
            f"Dropout output should be equal to input when training=False.\n"
            f"Input:\n{self.input}\n"
            f"Output:\n{output}"
        )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_p_0_case(self):
        """
        Feature: Dropout
        Description: Test drop_prob_0 Case: drop_prob=0.0,  training=True
        Exception: AssertionError
        """
        self.dropout.set_train(True)
        output = self.dropout_0(self.input)
        assert ms.ops.equal(output, self.input).all(), (
            f"Dropout output should be equal to input when training=False.\n"
            f"Input:\n{self.input}\n"
            f"Output:\n{output}"
        )
