#  Copyright 2025 Huawei Technologies Co., Ltd
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""Test save/load common info."""

import os
import unittest
from mindformers.checkpoint.checkpoint import CommonInfo


class TestCommonInfo(unittest.TestCase):
    """a test class for common json"""
    def test_save_load_common_info(self):
        """Test save and load common info."""
        common_info = CommonInfo()
        common_info['step_num'] = 1
        common_info['epoch_num'] = 2
        common_info['global_step'] = 3
        common_info['loss_scale'] = 4
        common_info['global_batch_size'] = 5
        common_filename = "./common.json"
        common_info.save_common(common_filename)
        self.assertTrue(os.path.exists(common_filename))

        common_info2 = CommonInfo()
        loaded_common_info = common_info2.load_common(common_filename)
        self.assertEqual(loaded_common_info['step_num'], 1)
        self.assertEqual(loaded_common_info['epoch_num'], 2)
        self.assertEqual(loaded_common_info['global_step'], 3)
        self.assertEqual(loaded_common_info['loss_scale'], 4)
        self.assertEqual(loaded_common_info['global_batch_size'], 5)

        os.remove(common_filename)
