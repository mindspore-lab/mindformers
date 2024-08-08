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
Test cogvlm2-video predict.
How to run this:
    pytest tests/st/test_model/test_cogvlm2_model/test_predict.py
"""
import os
import numpy as np
import pytest
from mindspore import Tensor
from mindspore import dtype as mstype
from mindformers.tools.register import MindFormerConfig
from mindformers.models import build_network
from mindformers.core.context.build_context import build_context


local_dir = os.path.dirname(os.path.realpath(__file__))


class TestCogVLM2VideoPredict:
    """A test class for testing model prediction."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_base_model(self):
        """
        Feature: Base model predict
        Description: Test base model prediction.
        Expectation: AssertionError
        """
        os.environ['USE_ROPE_SELF_DEFINE'] = 'True'
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        config = MindFormerConfig(f"{local_dir}/predict.yaml")
        build_context(config)
        network = build_network(config.model)
        input_ids = np.random.randint(0, 128, size=(1, 1280), dtype=np.int32)
        images = Tensor(np.random.random(size=(1, 3, 224, 224)), dtype=mstype.float32)
        video_context_pos = Tensor(np.array([[[0, i + 3] for i in range(66)]], dtype=np.int32))
        position_ids = Tensor(np.arange(2048, dtype=np.int32)).expand_dims(axis=0)
        _ = network.generate(input_ids=input_ids,
                             images=images,
                             video_context_pos=video_context_pos,
                             position_ids=position_ids)
