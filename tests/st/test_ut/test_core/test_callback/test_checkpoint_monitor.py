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
"""Test test_checkpoint_monitor.py"""
import unittest
import shutil
import tempfile
import pytest
from mindspore import ModelCheckpoint
from mindformers.core.callback.callback import CheckpointMonitor

class TestCheckpointMonitor(unittest.TestCase):
    """Test cases for CheckpointMonitor class"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_global_batch_size_parameter(self):
        """Test global batch size parameter"""
        monitor = CheckpointMonitor(
            global_batch_size=64
        )

        self.assertEqual(monitor.global_batch_size, 64)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test invalid directory type
        with self.assertRaises(TypeError):
            CheckpointMonitor(directory=123)

        # Test invalid save_checkpoint_steps type
        with self.assertRaises(TypeError):
            CheckpointMonitor(save_checkpoint_steps="invalid")

        # Test invalid keep_checkpoint_max type
        with self.assertRaises(TypeError):
            CheckpointMonitor(keep_checkpoint_max="invalid")

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_inheritance_from_model_checkpoint(self):
        """Test that CheckpointMonitor properly inherits from ModelCheckpoint"""
        monitor = CheckpointMonitor()

        # Check that it's an instance of ModelCheckpoint
        self.assertIsInstance(monitor, ModelCheckpoint)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_remove_redundancy_parameter(self):
        """Test remove_redundancy parameter"""
        monitor = CheckpointMonitor(
            remove_redundancy=True
        )

        self.assertTrue(monitor.remove_redundancy)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_network_params_parameter(self):
        """Test save_network_params parameter"""
        monitor = CheckpointMonitor(
            save_network_params=True
        )

        self.assertTrue(monitor.save_network_params)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_save_trainable_params_parameter(self):
        """Test save_trainable_params parameter"""
        monitor = CheckpointMonitor(
            save_trainable_params=True
        )

        self.assertTrue(monitor.save_trainable_params)

if __name__ == '__main__':
    unittest.main()
