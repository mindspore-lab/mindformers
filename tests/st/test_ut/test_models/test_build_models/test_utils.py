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
Test module for testing models utils for mindformers.
"""
import unittest
from unittest.mock import MagicMock
import pytest

from mindformers.models.utils import check_use_3d_tensor_parallel_valid


# Mock helper functions and constants
class ParallelMode:
    AUTO_PARALLEL = "auto_parallel"


def check_fine_grain_interleave_valid(fine_grain):
    return fine_grain is not None and fine_grain > 1


class TestCheckUse3DTensorParallelValid(unittest.TestCase):
    """A class for testing CheckUse3DTensorParallelValid."""
    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_disabled_use_3d_tp(self):
        """Branch: use_3d_tensor_parallel = False → return False"""
        config = MagicMock()
        config.use_3d_tensor_parallel = False
        result = check_use_3d_tensor_parallel_valid(config)
        self.assertFalse(result)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_config_none(self):
        """Branch: config is None → return False"""
        result = check_use_3d_tensor_parallel_valid(None)
        self.assertFalse(result)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_parallel_config_none(self):
        """Branch: config.parallel_config is None → return False"""
        config = MagicMock()
        config.use_3d_tensor_parallel = True
        config.parallel_config = None
        result = check_use_3d_tensor_parallel_valid(config)
        self.assertFalse(result)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_use_flash_attention_false(self):
        """Raise: use_flash_attention must be True"""
        config = self._create_valid_config()
        config.use_flash_attention = False
        with self.assertRaises(ValueError, msg="use_flash_attention must be True"):
            check_use_3d_tensor_parallel_valid(config)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_ulysses_cp_num_gt_1(self):
        """Raise: ulysses cp > 1 not supported"""
        config = self._create_valid_config()
        config.parallel_config.get_ulysses_cp_num.return_value = 2
        with self.assertRaises(ValueError, msg="ulysses cp must be 1"):
            check_use_3d_tensor_parallel_valid(config)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_moe_enabled(self):
        """Raise: MoE not supported"""
        config = self._create_valid_config()
        moe_mock = MagicMock()
        moe_mock.expert_num = 8
        config.moe_config = moe_mock
        with self.assertRaises(ValueError, msg="MoE not supported"):
            check_use_3d_tensor_parallel_valid(config)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_seq_parallel_false(self):
        """Raise: use_seq_parallel must be True"""
        config = self._create_valid_config()
        config.parallel_config.use_seq_parallel = False
        with self.assertRaises(ValueError, msg="use_seq_parallel must be True"):
            check_use_3d_tensor_parallel_valid(config)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_fine_grain_interleave_invalid(self):
        """Raise: fine_grain_interleave not supported"""
        config = self._create_valid_config()
        config.fine_grain_interleave = 2  # triggers True in check_fine_grain_interleave_valid
        with self.assertRaises(ValueError, msg="fine_grain_interleave not supported"):
            check_use_3d_tensor_parallel_valid(config)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_tp_product_mismatch(self):
        """Raise: tp_x * tp_y * tp_z != model_parallel"""
        config = self._create_valid_config()
        config.tp_x = 2
        config.tp_y = 2
        config.tp_z = 2
        config.parallel_config.model_parallel = 7  # 2*2*2=8 ≠ 7
        with self.assertRaises(ValueError, msg="tp product mismatch"):
            check_use_3d_tensor_parallel_valid(config)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def _create_valid_config(self):
        """Helper to create a config that passes initial checks"""
        config = MagicMock()
        config.use_3d_tensor_parallel = True
        config.use_flash_attention = True
        config.fine_grain_interleave = None  # valid
        config.moe_config = None

        parallel_config = MagicMock()
        parallel_config.get_ulysses_cp_num.return_value = 1
        parallel_config.use_seq_parallel = True
        parallel_config.model_parallel = 4
        config.parallel_config = parallel_config

        return config
