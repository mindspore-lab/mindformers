#  Copyright 2025 Huawei Technologies Co., Ltd
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""Test for transform_checkpoint.py"""
# pylint: disable=W0212
import os
from unittest.mock import patch, MagicMock

import pytest
from mindformers.tools.ckpt_transform import transform_checkpoint
from mindformers.tools.ckpt_transform.transform_checkpoint import TransformCkpt, main


class TestTransformCkpt:
    """Test TransformCkpt class"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init(self):
        """Test __init__ method"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            assert transform_ckpt.world_size == 1
            assert transform_ckpt.rank_id == 0
            assert transform_ckpt.is_main_rank is True
            assert transform_ckpt.npu_num_per_node == 1
            assert transform_ckpt.transform_process_num == 1
            assert transform_ckpt.transform_by_rank is False

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_transform_rank_id_list(self):
        """Test _get_transform_rank_id_list method"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=8), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=8):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=8,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=8
            )
            rank_list = transform_ckpt._get_transform_rank_id_list(2)
            assert rank_list == [0, 4]

            rank_list = transform_ckpt._get_transform_rank_id_list(8)
            assert rank_list == [0, 1, 2, 3, 4, 5, 6, 7]

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_strategy_file(self, tmp_path):
        """Test get_strategy method with file"""
        # Create test ckpt file
        test_ckpt_path = os.path.join(tmp_path, "test.ckpt")
        with open(test_ckpt_path, "w", encoding="utf-8") as f:
            f.write("test ckpt content")

        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            strategy_path = transform_ckpt.get_strategy(test_ckpt_path)
            assert strategy_path == test_ckpt_path

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_strategy_none(self):
        """Test get_strategy method with None"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            strategy_path = transform_ckpt.get_strategy(None)
            assert strategy_path is None

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_src_checkpoint_and_strategy_invalid(self, tmp_path):
        """Test check_src_checkpoint_and_strategy method with invalid input"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            with pytest.raises(ValueError):
                transform_ckpt.check_src_checkpoint_and_strategy(tmp_path, None)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_soft_link_of_checkpoint(self, tmp_path):
        """Test build_soft_link_of_checkpoint method with various input types"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )

            # Test 1: Invalid directory (no rank_0 folder or ckpt files)
            invalid_dir = os.path.join(tmp_path, "invalid_dir")
            os.makedirs(invalid_dir)
            soft_link_dir = os.path.join(tmp_path, "soft_link1")
            os.makedirs(soft_link_dir)

            with pytest.raises(ValueError):
                transform_ckpt.build_soft_link_of_checkpoint(invalid_dir, soft_link_dir)

            # Test 2: File input (ckpt file)
            test_ckpt_path = os.path.join(tmp_path, "test.ckpt")
            with open(test_ckpt_path, "w", encoding="utf-8") as f:
                f.write("test ckpt content")

            soft_link_dir = os.path.join(tmp_path, "soft_link2")
            os.makedirs(soft_link_dir)

            with patch("mindformers.tools.ckpt_transform.transform_checkpoint."
                       "make_soft_link") as mock_make_soft_link:
                transform_ckpt.build_soft_link_of_checkpoint(test_ckpt_path, soft_link_dir)
                mock_make_soft_link.assert_called_once()

            # Test 3: Directory with rank_0 folder
            valid_dir = os.path.join(tmp_path, "valid_dir")
            rank_0_dir = os.path.join(valid_dir, "rank_0")
            os.makedirs(rank_0_dir)
            valid_ckpt = os.path.join(rank_0_dir, "test.ckpt")
            with open(valid_ckpt, "w", encoding="utf-8") as f:
                f.write("valid ckpt content")

            soft_link_dir = os.path.join(tmp_path, "soft_link3")
            os.makedirs(soft_link_dir)

            with patch("mindformers.tools.ckpt_transform.transform_checkpoint."
                       "make_soft_link") as mock_make_soft_link:
                transform_ckpt.build_soft_link_of_checkpoint(valid_dir, soft_link_dir)
                mock_make_soft_link.assert_called_once()

            # Test 4: Directory with ckpt files directly
            ckpt_dir = os.path.join(tmp_path, "ckpt_dir")
            os.makedirs(ckpt_dir)
            ckpt1 = os.path.join(ckpt_dir, "ckpt1.ckpt")
            ckpt2 = os.path.join(ckpt_dir, "ckpt2.ckpt")
            with open(ckpt1, "w", encoding="utf-8") as f:
                f.write("ckpt1 content")
            with open(ckpt2, "w", encoding="utf-8") as f:
                f.write("ckpt2 content")

            soft_link_dir = os.path.join(tmp_path, "soft_link4")
            os.makedirs(soft_link_dir)

            with patch("mindformers.tools.ckpt_transform.transform_checkpoint."
                       "make_soft_link") as mock_make_soft_link:
                transform_ckpt.build_soft_link_of_checkpoint(ckpt_dir, soft_link_dir)
                # Should be called twice, once for each ckpt file
                assert mock_make_soft_link.call_count == 2

            # Test 5: Directory with both rank folders and ckpt files
            mixed_dir = os.path.join(tmp_path, "mixed_dir")
            mixed_rank_0_dir = os.path.join(mixed_dir, "rank_0")
            os.makedirs(mixed_rank_0_dir)
            mixed_ckpt = os.path.join(mixed_rank_0_dir, "mixed.ckpt")
            with open(mixed_ckpt, "w", encoding="utf-8") as f:
                f.write("mixed ckpt content")

            # Add a direct ckpt file in mixed_dir
            direct_ckpt = os.path.join(mixed_dir, "direct.ckpt")
            with open(direct_ckpt, "w", encoding="utf-8") as f:
                f.write("direct ckpt content")

            soft_link_dir = os.path.join(tmp_path, "soft_link5")
            os.makedirs(soft_link_dir)

            with patch("mindformers.tools.ckpt_transform.transform_checkpoint."
                       "make_soft_link") as mock_make_soft_link:
                transform_ckpt.build_soft_link_of_checkpoint(mixed_dir, soft_link_dir)
                # Should be called once for the rank folder, ignoring the direct ckpt file
                mock_make_soft_link.assert_called_once()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_clear_cache(self, tmp_path):
        """Test clear_cache method"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            # Add a cache file
            cache_file = os.path.join(tmp_path, "cache.txt")
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write("cache content")
            transform_ckpt.cache_list.append(cache_file)
            # Clear cache
            with patch("mindformers.tools.ckpt_transform.transform_checkpoint."
                       "delete_file") as mock_delete_file:
                transform_ckpt.clear_cache()
                mock_delete_file.assert_called_once_with(cache_file)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transform_checkpoints(self, tmp_path):
        """Test transform_checkpoints method"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms") as mock_ms, \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            dst_ckpt_dir = os.path.join(tmp_path, "dst_ckpt")
            transform_ckpt.transform_checkpoints(
                src_checkpoint=tmp_path,
                dst_checkpoint=dst_ckpt_dir,
                prefix="checkpoint_",
                src_strategy=None,
                dst_strategy=None
            )
            mock_ms.transform_checkpoints.assert_called_once()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transform_checkpoint_by_rank(self, tmp_path):
        """Test transform_checkpoint_by_rank method"""
        # Create test ckpt file
        test_ckpt_path = os.path.join(tmp_path, "test.ckpt")
        with open(test_ckpt_path, "w", encoding="utf-8") as f:
            f.write("test ckpt content")

        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=8), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms") as mock_ms, \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.glob", return_value=[test_ckpt_path]), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=8):
            # Mock rank_list_for_transform to return a list
            mock_ms.rank_list_for_transform.return_value = [0]
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=8,
                transform_process_num=1,
                transform_by_rank=True,
                npu_num_per_node=8
            )
            dst_ckpt_dir = os.path.join(tmp_path, "dst_ckpt")
            transform_ckpt.transform_checkpoint_by_rank(
                src_checkpoint=tmp_path,
                dst_checkpoint=dst_ckpt_dir,
                prefix="checkpoint_",
                src_strategy=None,
                dst_strategy=None
            )
            # Check that transform_checkpoint_by_rank was called 8 times (once for each rank)
            assert mock_ms.transform_checkpoint_by_rank.call_count == 8

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_call(self, tmp_path):
        """Test __call__ method"""
        # Create test ckpt file
        test_ckpt_path = os.path.join(tmp_path, "test.ckpt")
        with open(test_ckpt_path, "w", encoding="utf-8") as f:
            f.write("test ckpt content")

        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.barrier_world"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.remake_folder"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.make_soft_link"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=False), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            # Mock get_strategy to return None
            with patch.object(transform_ckpt, "get_strategy", return_value=None), \
                    patch.object(transform_ckpt, "transform_ckpt"), \
                    patch.object(transform_ckpt, "clear_cache"), \
                    patch("os.listdir", return_value=[]):
                result = transform_ckpt(
                    src_checkpoint=test_ckpt_path,
                    dst_checkpoint_dir=None,
                    src_strategy=None,
                    dst_strategy=None,
                    prefix="checkpoint_"
                )
                assert result is not None

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_call_auto_trans_ckpt_true(self, tmp_path):
        """Test __call__ method with auto_trans_ckpt=True"""
        # Create test ckpt file
        test_ckpt_path = os.path.join(tmp_path, "test.ckpt")
        with open(test_ckpt_path, "w", encoding="utf-8") as f:
            f.write("test ckpt content")

        # Test with auto_trans_ckpt=True and world_size>1
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=2), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.barrier_world"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.remake_folder"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.make_soft_link"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=False), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms.get_auto_parallel_context",
                      return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_output_root_path",
                      return_value=str(tmp_path)), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            # Create dst_strategy_dir
            dst_strategy_dir = os.path.join(tmp_path, "strategy")
            os.makedirs(dst_strategy_dir, exist_ok=True)

            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=True,
                rank_id=0,
                world_size=2,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )

            # Mock get_strategy to return a strategy file
            strategy_file = os.path.join(dst_strategy_dir,
                                         "test_strategy_rank_0.ckpt")
            with open(strategy_file, "w", encoding="utf-8") as f:
                f.write("test strategy")

            with patch.object(transform_ckpt, "get_strategy", return_value=strategy_file), \
                    patch.object(transform_ckpt, "get_dst_strategy", return_value=strategy_file), \
                    patch.object(transform_ckpt, "transform_ckpt"), \
                    patch.object(transform_ckpt, "clear_cache"), \
                    patch("os.listdir", return_value=[]):
                result = transform_ckpt(
                    src_checkpoint=test_ckpt_path,
                    dst_checkpoint_dir=None,
                    src_strategy=None,
                    dst_strategy=None,
                    prefix="checkpoint_"
                )
                assert result is not None

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_call_modelarts(self, tmp_path):
        """Test __call__ method with ModelArts environment"""
        # Create test ckpt file
        test_ckpt_path = os.path.join(tmp_path, "test.ckpt")
        with open(test_ckpt_path, "w", encoding="utf-8") as f:
            f.write("test ckpt content")

        # Import the module and add mox attribute directly
        mock_mox = MagicMock()
        mock_mox.file = MagicMock()
        mock_mox.file.exists.return_value = True
        transform_checkpoint.mox = mock_mox

        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.barrier_world"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.remake_folder"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.make_soft_link"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms.get_auto_parallel_context",
                      return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_output_root_path",
                      return_value=str(tmp_path)), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_remote_save_url",
                      return_value="s3://bucket/path"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=True,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )

            # Mock get_strategy to return None
            with patch.object(transform_ckpt, "get_strategy", return_value=None), \
                    patch.object(transform_ckpt, "get_dst_strategy", return_value=None), \
                    patch.object(transform_ckpt, "transform_ckpt"), \
                    patch.object(transform_ckpt, "clear_cache"), \
                    patch("os.listdir", return_value=[]):
                result = transform_ckpt(
                    src_checkpoint=test_ckpt_path,
                    dst_checkpoint_dir=None,
                    src_strategy=None,
                    dst_strategy=None,
                    prefix="checkpoint_"
                )
                assert result is not None

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_src_checkpoint_and_strategy_valid(self, tmp_path):
        """Test check_src_checkpoint_and_strategy method with valid input"""
        # Create test ckpt file
        test_ckpt_path = os.path.join(tmp_path, "test.ckpt")
        with open(test_ckpt_path, "w", encoding="utf-8") as f:
            f.write("test ckpt content")

        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            # Create a valid directory structure
            valid_dir = os.path.join(tmp_path, "valid_ckpt")
            rank_0_dir = os.path.join(valid_dir, "rank_0")
            os.makedirs(rank_0_dir, exist_ok=True)
            valid_ckpt = os.path.join(rank_0_dir, "test.ckpt")
            with open(valid_ckpt, "w", encoding="utf-8") as f:
                f.write("valid ckpt content")

            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            # This should not raise an exception
            transform_ckpt.check_src_checkpoint_and_strategy(valid_dir, test_ckpt_path)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transform_ckpt(self, tmp_path):
        """Test transform_ckpt method with various scenarios"""
        # Test 1: Both src_strategy and dst_strategy are None
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )

            # Create a valid directory structure
            valid_dir = os.path.join(tmp_path, "valid_ckpt")
            rank_0_dir = os.path.join(valid_dir, "rank_0")
            os.makedirs(rank_0_dir, exist_ok=True)
            valid_ckpt = os.path.join(rank_0_dir, "test.ckpt")
            with open(valid_ckpt, "w", encoding="utf-8") as f:
                f.write("valid ckpt content")

            # This should raise ValueError since both strategies are None
            with pytest.raises(ValueError):
                transform_ckpt.transform_ckpt(
                    src_checkpoint=valid_dir,
                    dst_checkpoint_dir=tmp_path,
                    src_strategy=None,
                    dst_strategy=None,
                    prefix="checkpoint_"
                )

        # Test 2: transform_ckpt with exception handling
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=False), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.create_file") as mock_create_file, \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )

            # Create a valid directory structure
            valid_dir = os.path.join(tmp_path, "valid_ckpt")
            rank_0_dir = os.path.join(valid_dir, "rank_0")
            os.makedirs(rank_0_dir, exist_ok=True)
            valid_ckpt = os.path.join(rank_0_dir, "test.ckpt")
            with open(valid_ckpt, "w", encoding="utf-8") as f:
                f.write("valid ckpt content")

            # Mock transform_checkpoints to raise an exception
            with patch.object(transform_ckpt, "check_src_checkpoint_and_strategy"), \
                    patch.object(transform_ckpt, "transform_checkpoints",
                                 side_effect=Exception("Transform failed")), \
                    patch.object(transform_ckpt, "wait_transform"):
                transform_ckpt.transform_ckpt(
                    src_checkpoint=valid_dir,
                    dst_checkpoint_dir=tmp_path,
                    src_strategy="src_strategy.ckpt",
                    dst_strategy="dst_strategy.ckpt",
                    prefix="checkpoint_"
                )
                # Check that transform_failed file was created
                mock_create_file.assert_called()

        # Test 3: transform_ckpt with ModelArts case
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_remote_save_url",
                      return_value="s3://bucket/"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_output_root_path",
                      return_value="/tmp/"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.create_file") as mock_create_file, \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=True,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )

            # Create a valid directory structure
            valid_dir = os.path.join(tmp_path, "valid_ckpt")
            rank_0_dir = os.path.join(valid_dir, "rank_0")
            os.makedirs(rank_0_dir, exist_ok=True)
            valid_ckpt = os.path.join(rank_0_dir, "test.ckpt")
            with open(valid_ckpt, "w", encoding="utf-8") as f:
                f.write("valid ckpt content")

                # Mock transform_checkpoints to succeed
            with patch.object(transform_ckpt, "check_src_checkpoint_and_strategy"), \
                    patch.object(transform_ckpt, "transform_checkpoints"), \
                    patch.object(transform_ckpt, "wait_transform"), \
                    patch.object(transform_ckpt, "send_transformed_checkpoint_to_obs"):
                transform_ckpt.transform_ckpt(
                    src_checkpoint=valid_dir,
                    dst_checkpoint_dir=tmp_path,
                    src_strategy="src_strategy.ckpt",
                    dst_strategy="dst_strategy.ckpt",
                    prefix="checkpoint_"
                )
                # Check that transform_succeed file was created
                mock_create_file.assert_called()

            # Test 4, \ transform_ckpt when rank_id is not in transform_rank_id_list
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=2), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=False), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=1,
                world_size=2,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            # Set transform_rank_id_list to [0] so rank 1 is not in the list
            transform_ckpt.transform_rank_id_list = [0]

            # Create a valid directory structure
            valid_dir = os.path.join(tmp_path, "valid_ckpt")
            rank_0_dir = os.path.join(valid_dir, "rank_0")
            os.makedirs(rank_0_dir, exist_ok=True)
            valid_ckpt = os.path.join(rank_0_dir, "test.ckpt")
            with open(valid_ckpt, "w", encoding="utf-8") as f:
                f.write("valid ckpt content")

            # Mock wait_transform to avoid infinite loop
            with patch.object(transform_ckpt, "check_src_checkpoint_and_strategy"), \
                    patch.object(transform_ckpt, "wait_transform"):
                transform_ckpt.transform_ckpt(
                    src_checkpoint=valid_dir,
                    dst_checkpoint_dir=tmp_path,
                    src_strategy="src_strategy.ckpt",
                    dst_strategy="dst_strategy.ckpt",
                    prefix="checkpoint_"
                )
                # Should complete without calling transform_checkpoints

        # Test 5: transform_ckpt with transform_by_rank=True
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=False), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=True,
                npu_num_per_node=1
            )

            # Create a valid directory structure
            valid_dir = os.path.join(tmp_path, "valid_ckpt")
            rank_0_dir = os.path.join(valid_dir, "rank_0")
            os.makedirs(rank_0_dir, exist_ok=True)
            valid_ckpt = os.path.join(rank_0_dir, "test.ckpt")
            with open(valid_ckpt, "w", encoding="utf-8") as f:
                f.write("valid ckpt content")

            # Mock transform_checkpoint_by_rank to succeed
            with patch.object(transform_ckpt, "check_src_checkpoint_and_strategy"), \
                    patch.object(transform_ckpt, "transform_checkpoint_by_rank"), \
                    patch.object(transform_ckpt, "wait_transform"):
                transform_ckpt.transform_ckpt(
                    src_checkpoint=valid_dir,
                    dst_checkpoint_dir=tmp_path,
                    src_strategy="src_strategy.ckpt",
                    dst_strategy="dst_strategy.ckpt",
                    prefix="checkpoint_"
                )
                # Should complete successfully

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_invalid_npu_num(self):
        """Test __init__ method with invalid npu_num_per_node"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=2), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                pytest.raises(ValueError), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=2):
            TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=2,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=3  # Not a power of 2
            )

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_init_auto_trans_ckpt_true(self, tmp_path):
        """Test __init__ method with auto_trans_ckpt=True"""
        # Test with world_size=1 and auto_trans_ckpt=True
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms.get_auto_parallel_context",
                      return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_output_root_path",
                      return_value=str(tmp_path)), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=True,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            assert transform_ckpt.auto_trans_ckpt is True
            assert transform_ckpt.transformed_checkpoint_dir == os.path.join(tmp_path, "transformed_checkpoint")
            # No dst_strategy_dir when world_size=1
            assert not hasattr(transform_ckpt, 'dst_strategy_dir')

        # Test  world_size>1 and auto_trans_ckpt=True
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=2), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms.get_auto_parallel_context",
                      return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_output_root_path",
                      return_value=str(tmp_path)), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=True,
                rank_id=0,
                world_size=2,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            assert transform_ckpt.auto_trans_ckpt is True
            assert transform_ckpt.transformed_checkpoint_dir == os.path.join(tmp_path, "transformed_checkpoint")
            assert transform_ckpt.dst_strategy_dir == os.path.join(tmp_path, "strategy")

        # Test  pipeline parallelism and auto_trans_ckpt=True
        with  patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=2), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms.get_auto_parallel_context",
                      return_value=2), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_output_root_path",
                      return_value=str(tmp_path)), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=True,
                rank_id=0,
                world_size=2,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            assert transform_ckpt.use_pipeline is True

        # Test  ModelArts environment and auto_trans_ckpt=True
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=2), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms.get_auto_parallel_context",
                      return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_output_root_path",
                      return_value=str(tmp_path)), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_remote_save_url",
                      return_value="s3://bucket/path"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=True,
                rank_id=0,
                world_size=2,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            assert hasattr(transform_ckpt, 'transformed_checkpoint_dir_obs')
            assert hasattr(transform_ckpt, 'dst_strategy_dir_obs')

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_main(self):
        """Test main function"""
        with patch("sys.argv", [
            "transform_checkpoint.py",
            "--src_checkpoint", "/path/to/src/ckpt",
            "--dst_checkpoint_dir", "/path/to/dst/ckpt",
            "--src_strategy", "/path/to/src/strategy.ckpt",
            "--dst_strategy", "/path/to/dst/strategy.ckpt",
            "--prefix", "checkpoint_",
            "--rank_id", "0",
            "--world_size", "1",
            "--transform_process_num", "1"
            # 不传入transform_by_rank参数，使用默认值False
        ]), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.TransformCkpt") as mock_transform_ckpt:
            # Mock the TransformCkpt class and its __call__ method
            mock_instance = mock_transform_ckpt.return_value
            mock_instance.return_value = "/path/to/dst/ckpt"

            # Import and call main function
            main()

            # Verify TransformCkpt was initialized correctly
            mock_transform_ckpt.assert_called_once()
            _, kwargs = mock_transform_ckpt.call_args
            assert kwargs["rank_id"] == 0
            assert kwargs["world_size"] == 1
            assert kwargs["transform_process_num"] == 1
            assert not kwargs["transform_by_rank"]

            # Verify TransformCkpt instance was called correctly
            mock_instance.assert_called_once()
            _, call_kwargs = mock_instance.call_args
            assert call_kwargs["src_checkpoint"] == "/path/to/src/ckpt"
            assert call_kwargs["dst_checkpoint_dir"] == "/path/to/dst/ckpt"
            assert call_kwargs["src_strategy"] == "/path/to/src/strategy.ckpt"
            assert call_kwargs["dst_strategy"] == "/path/to/dst/strategy.ckpt"
            assert call_kwargs["prefix"] == "checkpoint_"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_transform_rank_id_list_invalid(self):
        """Test _get_transform_rank_id_list method with invalid input"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=8), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=8):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=8,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=8
            )
            # Test with transform_process_num < 1
            with pytest.raises(ValueError):
                transform_ckpt._get_transform_rank_id_list(0)
            # Test with transform_process_num not divisible by world_size
            with pytest.raises(ValueError):
                transform_ckpt._get_transform_rank_id_list(3)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_strategy(self, tmp_path):
        """Test get_strategy method with various inputs"""
        with (patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
              patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
              patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
              patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1)):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )

            # Test 1: None input
            result = transform_ckpt.get_strategy(None)
            assert result is None

            # Test 2: "None" string input
            result = transform_ckpt.get_strategy("None")
            assert result is None

            # Test 3: Invalid path
            invalid_path = os.path.join(tmp_path, "invalid_path")
            with pytest.raises(ValueError):
                transform_ckpt.get_strategy(invalid_path)

            # Test 4: File input
            test_file = os.path.join(tmp_path, "test_strategy.ckpt")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("test strategy content")

            result = transform_ckpt.get_strategy(test_file)
            assert result == test_file

            # Test 5: Directory input with main rank
            strategy_dir = os.path.join(tmp_path, "strategy_dir")
            os.makedirs(strategy_dir)

            # Create a strategy file in the directory
            strategy_file = os.path.join(strategy_dir, "strategy_0.ckpt")
            with open(strategy_file, "w", encoding="utf-8") as f:
                f.write("strategy content")

            # Mock ms.merge_pipeline_strategys
            with patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms."
                       "merge_pipeline_strategys") as mock_merge, \
                    patch("mindformers.tools.ckpt_transform.transform_checkpoint.create_file") as mock_create_file:
                result = transform_ckpt.get_strategy(strategy_dir)
                expected_merge_path = os.path.join(strategy_dir, "merged_ckpt_strategy.ckpt")
                assert result == expected_merge_path
                mock_merge.assert_called_once_with(strategy_dir, expected_merge_path)
                mock_create_file.assert_called_once()

            # Test 6: Directory input with main rank and existing merged strategy
            # Create merged strategy file
            merged_strategy = os.path.join(strategy_dir, "merged_ckpt_strategy.ckpt")
            with open(merged_strategy, "w", encoding="utf-8") as f:
                f.write("merged strategy content")

            # Mock ms.merge_pipeline_strategys
            with patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms."
                       "merge_pipeline_strategys") as mock_merge, \
                    patch("mindformers.tools.ckpt_transform.transform_checkpoint.create_file") as mock_create_file, \
                    patch("os.remove") as mock_remove:
                result = transform_ckpt.get_strategy(strategy_dir)
                expected_merge_path = os.path.join(strategy_dir, "merged_ckpt_strategy.ckpt")
                assert result == expected_merge_path
                mock_remove.assert_called_once_with(expected_merge_path)
                mock_merge.assert_called_once_with(strategy_dir, expected_merge_path)
                mock_create_file.assert_called_once()

            # Test 7: Directory input with non-main rank
            with patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=False):
                transform_ckpt_non_main = TransformCkpt(
                    auto_trans_ckpt=False,
                    rank_id=1,
                    world_size=2,
                    transform_process_num=1,
                    transform_by_rank=False,
                    npu_num_per_node=1
                )

                # Create merged_succeed.txt to avoid infinite loop
                merged_succeed_txt = os.path.join(strategy_dir, "merge_succeed.txt")
                with open(merged_succeed_txt, "w", encoding="utf-8") as f:
                    f.write("merge succeed")

                result = transform_ckpt_non_main.get_strategy(strategy_dir)
                expected_merge_path = os.path.join(strategy_dir, "merged_ckpt_strategy.ckpt")
                assert result == expected_merge_path

            # Test 8: Directory input with rank_id parameter
            with patch("mindformers.tools.ckpt_transform.transform_checkpoint.ms."
                       "merge_pipeline_strategys") as mock_merge, \
                    patch("mindformers.tools.ckpt_transform.transform_checkpoint.create_file") as mock_create_file:
                result = transform_ckpt.get_strategy(strategy_dir, rank_id=1)
                expected_merge_path = os.path.join(strategy_dir, "merged_ckpt_strategy_by_rank_1.ckpt")
                assert result == expected_merge_path
                mock_merge.assert_called_once_with(strategy_dir, expected_merge_path)
                mock_create_file.assert_called_once()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_build_soft_link_of_checkpoint_invalid_file(self, tmp_path):
        """Test build_soft_link_of_checkpoint method with invalid file"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            # Create an invalid file (not a ckpt file)
            invalid_file = os.path.join(tmp_path, "invalid.txt")
            with open(invalid_file, "w", encoding="utf-8") as f:
                f.write("invalid content")
            soft_link_dir = os.path.join(tmp_path, "soft_link")
            os.makedirs(soft_link_dir)
            with pytest.raises(ValueError):
                transform_ckpt.build_soft_link_of_checkpoint(invalid_file, soft_link_dir)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_send_strategy_to_obs(self, tmp_path):
        # pylint: disable=W0613
        """Test send_strategy_to_obs method"""
        # Create mock functions for mox.file operations
        def mock_copy(*args, **kwargs):
            return None

        def mock_exists(*args, **kwargs):
            return False

        # Mock the moxing module and mox alias
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_remote_save_url",
                      return_value="s3://bucket"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.mox", create=True) as mock_mox, \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            # Configure the mock mox object
            mock_mox.file.copy = mock_copy
            mock_mox.file.exists = mock_exists

            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=True,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            # Add required attributes for ModelArts
            transform_ckpt.dst_strategy_dir_obs = "s3://bucket/strategy"

            # Create a strategy file
            strategy_file = os.path.join(tmp_path, "test_strategy.ckpt")
            with open(strategy_file, "w", encoding="utf-8") as f:
                f.write("test strategy content")

            transform_ckpt.send_strategy_to_obs(strategy_file)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_send_transformed_checkpoint_to_obs(self, tmp_path):
        """Test send_transformed_checkpoint_to_obs method"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_remote_save_url",
                      return_value="s3://bucket"), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.mox_adapter") as mock_mox_adapter, \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=True,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            # Add required attributes for ModelArts
            transform_ckpt.transformed_checkpoint_dir_obs = "s3://bucket/transformed"

            # Create a dst checkpoint directory
            dst_ckpt_dir = os.path.join(tmp_path, "dst_ckpt")
            os.makedirs(dst_ckpt_dir)

            transform_ckpt.send_transformed_checkpoint_to_obs(dst_ckpt_dir)
            mock_mox_adapter.assert_called_once()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_wait_transform(self, tmp_path):
        """Test wait_transform method"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=False), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )

            # Create a ckpt_dir
            ckpt_dir = os.path.join(tmp_path, "ckpt_dir")
            os.makedirs(ckpt_dir)

            # Create transform_succeed file
            succeed_file = os.path.join(ckpt_dir, "transform_succeed_rank_0.txt")
            with open(succeed_file, "w", encoding="utf-8") as f:
                f.write("transform succeed")

            # This should return immediately since the succeed file exists
            transform_ckpt.wait_transform(ckpt_dir)

            # Test with transform_failed file
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=False):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )

            # Create a ckpt_dir
            ckpt_dir = os.path.join(tmp_path, "ckpt_dir_failed")
            os.makedirs(ckpt_dir)

            # Create transform_failed file
            failed_file = os.path.join(ckpt_dir, "transform_failed_rank_0.txt")
            with open(failed_file, "w", encoding="utf-8") as f:
                f.write("transform failed")

            # This should raise ValueError since a failed file exists
            with pytest.raises(ValueError):
                transform_ckpt.wait_transform(ckpt_dir)

            # Test with ModelArts case
            # Import the module and add mox attribute directly

            mock_mox = MagicMock()
            mock_mox.file = MagicMock()

            # Define a side_effect to return different results based on the pattern
            def mock_glob_side_effect(pattern):
                if 'transform_failed' in pattern:
                    return []  # No failed files
                if 'transform_succeed' in pattern:
                    return ["s3://bucket/path/transformed_checkpoint/ckpt_dir_modelarts/transform_succeed_rank_0.txt"]
                return []

            mock_mox.file.glob.side_effect = mock_glob_side_effect
            transform_checkpoint.mox = mock_mox

            with patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=True), \
                    patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_remote_save_url",
                          return_value="s3://bucket/path"), \
                    patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_output_root_path",
                          return_value=str(tmp_path)):
                # Create TransformCkpt instance
                transform_ckpt = TransformCkpt(
                    auto_trans_ckpt=True,
                    rank_id=0,
                    world_size=1,
                    transform_process_num=1,
                    transform_by_rank=False,
                    npu_num_per_node=1
                )

                # Create a ckpt_dir
                ckpt_dir = os.path.join(tmp_path, "ckpt_dir_modelarts")
                os.makedirs(ckpt_dir)

                # This should return immediately since mock returns succeed file
                transform_ckpt.wait_transform(ckpt_dir)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_wait_collect_all_strategy(self, tmp_path):
        """Test wait_collect_all_strategy method"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.check_in_modelarts", return_value=False), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=True,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            # Add required attributes
            transform_ckpt.dst_strategy_dir = tmp_path

            # Create a strategy file
            strategy_file = os.path.join(tmp_path, "ckpt_strategy_rank_0.ckpt")
            with open(strategy_file, "w", encoding="utf-8") as f:
                f.write("test strategy content")

            # This should return immediately since the strategy file exists
            transform_ckpt.wait_collect_all_strategy()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_clear_cache_not_main_rank(self, tmp_path):
        """Test clear_cache method when not main rank"""
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=False), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=1,
                world_size=2,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            # Add a cache file
            cache_file = os.path.join(tmp_path, "cache.txt")
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write("cache content")
            transform_ckpt.cache_list.append(cache_file)
            # Clear cache - should not delete anything since not main rank
            with patch("mindformers.tools.ckpt_transform.transform_checkpoint.delete_file") as mock_delete_file:
                transform_ckpt.clear_cache()
                mock_delete_file.assert_not_called()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_dst_strategy(self, tmp_path):
        """Test get_dst_strategy method"""
        # Test with world_size=1
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=1), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=1,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            result = transform_ckpt.get_dst_strategy("test_strategy.ckpt")
            assert result is None

        # Test with world_size > 1 and invalid dst_strategy
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=2), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=2,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            # Test with invalid dst_strategy (wrong rank suffix)
            with pytest.raises(ValueError):
                transform_ckpt.get_dst_strategy("test_strategy_rank_1.ckpt")

        # Test with world_size > 1 and valid dst_strategy
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=2), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            # Create a valid strategy file
            valid_strategy = os.path.join(tmp_path, "test_strategy_rank_0.ckpt")
            with open(valid_strategy, "w", encoding="utf-8") as f:
                f.write("valid strategy")

            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=2,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            transform_ckpt.use_pipeline = False
            result = transform_ckpt.get_dst_strategy(valid_strategy)
            assert result == valid_strategy

        # Test with pipeline parallelism and main rank
        with patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_group_size", return_value=2), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_real_rank", return_value=0), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.is_main_rank", return_value=True), \
                patch("mindformers.tools.ckpt_transform.transform_checkpoint.get_device_num_per_node", return_value=1):
            # Create a valid strategy file
            valid_strategy = os.path.join(tmp_path, "test_strategy_rank_0.ckpt")
            with open(valid_strategy, "w", encoding="utf-8") as f:
                f.write("valid strategy")

            # Create dst_strategy_dir with merged strategy
            dst_strategy_dir = os.path.join(tmp_path, "strategy")
            os.makedirs(dst_strategy_dir)
            merged_strategy = os.path.join(dst_strategy_dir, "merged_ckpt_strategy.ckpt")
            with open(merged_strategy, "w", encoding="utf-8") as f:
                f.write("merged strategy")

            transform_ckpt = TransformCkpt(
                auto_trans_ckpt=False,
                rank_id=0,
                world_size=2,
                transform_process_num=1,
                transform_by_rank=False,
                npu_num_per_node=1
            )
            transform_ckpt.use_pipeline = True
            transform_ckpt.dst_strategy_dir = dst_strategy_dir

            with patch.object(transform_ckpt, "get_strategy", return_value=merged_strategy), \
                    patch.object(transform_ckpt, "wait_collect_all_strategy"):
                result = transform_ckpt.get_dst_strategy(valid_strategy)
                assert result == merged_strategy
