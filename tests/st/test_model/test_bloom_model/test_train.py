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
Test bloom train.
How to run this:
    pytest tests/st/test_model/test_bloom_model/test_train.py
"""
import pytest
import mindspore as ms
from tests.utils.model_tester import ModelTester

from .base_model import get_config, get_model

ms.set_context(mode=0)


class TestBloomTrain:
    """A test class for testing model training precision."""

    @pytest.mark.level1
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
        # if set default, cause Memory pool not enough by large alibi tensor
        model_config.seq_length = 1024

        loss_std = [31.927244, 31.562233, 30.998333, 30.172699, 29.056198,
                    27.678558, 26.111250, 24.342562, 22.492617, 20.694494,
                    19.059685, 17.722950, 16.582079, 15.667753, 14.978457,
                    14.505189, 14.146716, 13.990182, 13.862482, 13.853906]

        model = get_model(model_config)

        runner.set_train(model, model_config, loss_std=loss_std)
