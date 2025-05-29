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
Test glm2, glm3, glm4, glm32k train.
How to run this:
    pytest tests/st/test_model/test_glm2_model/test_train.py
"""
import pytest
import mindspore as ms
import mindspore.common.dtype as mstype

from tests.utils.model_tester import ModelTester
from .base_model import get_config, get_model

ms.set_context(mode=0)


class TestGLM2Train:
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
        runner = ModelTester(run_mode='train', batch_size=4, use_label=True, experiment_mode=False)

        model_config = get_config()

        loss_std = [11.282303, 11.300609, 11.281874, 11.287205, 11.283550,
                    11.265226, 11.283127, 11.279236, 11.277109, 11.262154,
                    11.273281, 11.270094, 11.263002, 11.278083, 11.277325,
                    11.279003, 11.250218, 11.273275, 11.267607, 11.284098]

        model = get_model(model_config)

        model_config.seq_length -= 1  # set for generate data
        runner.set_train(model, model_config, loss_std=loss_std)

class TestGLM32kTrain:
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
        runner = ModelTester(run_mode='train', batch_size=1, use_label=True, experiment_mode=False)

        model_config = get_config()

        # change config for glm32k
        model_config.batch_size = 1
        model_config.compute_dtype = mstype.float16
        model_config.max_decode_length = 512
        model_config.no_recompute_layers = [20]
        model_config.param_init_type = mstype.float16
        model_config.rope_ratio = 50
        model_config.seq_length = 8192

        loss_std = [11.281449, 11.277131, 11.258755, 11.228244, 11.221319,
                    11.169483, 11.167768, 11.159433, 11.118815, 11.120639,
                    11.088744, 11.097591, 11.083368, 11.077658, 11.074391,
                    11.067731, 11.056860, 11.063673, 11.067738, 11.075453]

        model = get_model(model_config)

        model_config.seq_length -= 1  # set for generate data
        runner.set_train(model, model_config, loss_std=loss_std)
