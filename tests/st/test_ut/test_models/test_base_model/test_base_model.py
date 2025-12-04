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
"""
Test base model
"""
import os
import tempfile
import pytest

from mindformers import MindFormerConfig
from mindformers.models.base_config import BaseConfig
from mindformers.models.base_model import BaseModel

NUM_LAYERS = 1


class TestBaseModel:
    """A test class for testing model.save_pretrained() method."""

    def setup_method(self):
        """init test class."""
        with tempfile.TemporaryDirectory() as temp_dir_path:
            self.path = temp_dir_path

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_base_model(self):
        """
        Feature: Base_model save_pretrained()
        Description: Test llama save pretrained
        Expectation: Run successfully.
        """
        config = BaseConfig(num_layers=NUM_LAYERS)
        model = BaseModel(config)
        model.save_pretrained(self.path, save_name="mindspore_model")
        yaml_path = self.path + "/" + "mindspore_model.yaml"
        model_path = self.path + "/" + "mindspore_model.ckpt"
        assert os.path.exists(yaml_path)
        assert os.path.exists(model_path)

        mf_config = MindFormerConfig(yaml_path)
        assert mf_config.model.model_config.num_layers == NUM_LAYERS
        # pylint: disable=W0212
        model._get_config_args(pretrained_model_name_or_dir=self.path)
