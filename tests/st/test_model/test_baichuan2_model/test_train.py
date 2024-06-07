# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
Test baichuan2 train.
How to run this:
    pytest tests/st/test_model/test_baichuan2_model/test_train.py
"""
import pytest
import mindspore as ms
from tests.utils.model_tester import ModelTester

from .base_model import get_config, get_model

ms.set_context(mode=0)


class TestBaichuan2Train:
    """A test class for testing model training precision."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_base_model(self):
        """
        Feature: Base model train
        Description: Test base model training precision.
        Expectation: AssertionError
        """
        runner = ModelTester(run_mode='train', batch_size=4, experiment_mode=False)

        model_config = get_config()
        # if set 4096, cause Memory pool not enough by large alibi tensor
        model_config.seq_length = 512

        loss_std = [12.221302, 12.234898, 12.227418, 12.224919, 12.247095,
                    12.256886, 12.259717, 12.249476, 12.265904, 12.235092,
                    12.264286, 12.229508, 12.218564, 12.220242, 12.231214,
                    12.238742, 12.246398, 12.267619, 12.264290, 12.302188]

        model = get_model(model_config)

        runner.set_train(model, model_config, loss_std=loss_std)
