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
"""test common dataloader"""
import os
import unittest
import tempfile
import pytest
import mindspore
from mindformers import MindFormerConfig
from mindformers import CommonDataLoader

from tests.st.test_ut.test_dataset.get_test_data import get_llava_data


class TestCommonDataloader(unittest.TestCase):
    """A test class for testing Common dataloader"""

    @classmethod
    def setUpClass(cls):
        os.environ['USE_OM'] = "OFF"
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.path = cls.temp_dir.name
        get_llava_data(cls.path)
        cls.handler_config = MindFormerConfig(
            **{'type': "LlavaInstructDataHandler", "image_dir": cls.path, "output_columns": ["conversations"]}
        )

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_illegal_input(self):
        with pytest.raises(ValueError):
            assert CommonDataLoader(shuffle=False, input_columns=["conversations"], handler=self.handler_config)

        data_loader = CommonDataLoader(path=os.path.join(self.path, "text"), shuffle=False,
                                       input_columns=["conversations"], handler=self.handler_config, mock_input="")
        assert isinstance(data_loader, mindspore.dataset.GeneratorDataset)
