#  Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test checkpoint utils.py"""
import os
import tempfile
import unittest
from unittest import mock

from mindformers.checkpoint.utils import _check_checkpoint_path
from mindformers.checkpoint.utils import check_iteration_path_exists
from mindformers.checkpoint.utils import get_checkpoint_iter_dir
from mindformers.checkpoint.utils import get_checkpoint_tracker_filename
from mindformers.checkpoint.utils import get_common_filename
from mindformers.checkpoint.utils import get_metadata_filename


class TestCheckpointUtils(unittest.TestCase):
    """a test class for checkpoint utils"""

    def test_get_checkpoint_iter_dir(self):
        """a test case for get checkpoint iter dir"""
        base_path = "/tmp/checkpoints"
        iteration = 12345
        expected = os.path.join(base_path, "iter_00012345")
        self.assertEqual(get_checkpoint_iter_dir(base_path, iteration), expected)

        with self.assertRaises(ValueError):
            get_checkpoint_iter_dir(base_path, "not_an_int")

    def test_get_checkpoint_tracker_filename(self):
        """a test case for get checkpoint tracker filename"""
        base_path = "/tmp/checkpoints"
        expected = os.path.join(base_path, "latest_checkpointed_iteration.txt")
        self.assertEqual(get_checkpoint_tracker_filename(base_path), expected)

    def test_get_common_filename(self):
        """a test case for get common filename"""
        base_path = "/tmp/checkpoints"
        iteration = 12345
        expected = os.path.join(base_path, "iter_00012345", "common.json")
        self.assertEqual(get_common_filename(base_path, iteration), expected)

    def test_get_metadata_filename(self):
        """a test case for get metadata filename"""
        base_path = "/tmp/checkpoints"
        iteration = 12345
        expected = os.path.join(base_path, "iter_00012345", "metadata.json")
        self.assertEqual(get_metadata_filename(base_path, iteration), expected)

    @mock.patch("os.path.exists")
    @mock.patch("os.path.isdir")
    @mock.patch("os.path.isfile")
    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data="12345\n")
    def test_check_iteration_path_exists_valid(self, mock_isfile, mock_isdir, mock_exists):
        """a test case for check iteration path exists"""
        checkpoints_path = "/tmp/checkpoints"

        # 模拟 tracker 文件存在
        mock_isfile.return_value = True
        mock_exists.return_value = True
        mock_isdir.return_value = True

        result = check_iteration_path_exists(checkpoints_path)
        self.assertTrue(result)

    @mock.patch("os.path.isfile", return_value=False)
    def test_check_iteration_path_missing_tracker_file(self):
        """a test case for check iteration path with no tracker file"""
        checkpoints_path = "/tmp/checkpoints"
        with self.assertRaises(FileNotFoundError):
            check_iteration_path_exists(checkpoints_path)

    @mock.patch("os.path.isfile", return_value=True)
    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data="invalid\n")
    def test_check_iteration_path_invalid_iteration(self):
        """a test case for check iteration path with invalid iteration"""
        checkpoints_path = "/tmp/checkpoints"
        with self.assertRaises(ValueError):
            check_iteration_path_exists(checkpoints_path)

    def test_check_checkpoint_path_valid(self):
        """a test case for check valid path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir
            result = _check_checkpoint_path(path)
            self.assertEqual(result, path)

    def test_check_checkpoint_path_with_trailing_slash(self):
        """a test case for check valid path with slash"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + "/"
            result = _check_checkpoint_path(path)
            self.assertEqual(result, tmpdir)

    def test_check_checkpoint_path_invalid_type(self):
        """a test case for check valid path with invalid type"""
        with self.assertRaises(ValueError):
            _check_checkpoint_path(123)

    def test_check_checkpoint_path_not_exists(self):
        """a test case for check valid path with non-exist path"""
        with self.assertRaises(FileNotFoundError):
            _check_checkpoint_path("/non/existent/path")
