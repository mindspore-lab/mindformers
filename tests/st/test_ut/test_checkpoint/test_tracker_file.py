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
"""Test save/load tracker file."""

import os
import threading
import unittest
from unittest import mock
from mindformers.checkpoint.checkpoint import AsyncSaveManager


class TestAsyncSaveTrackerFile(unittest.TestCase):
    """a test class for async save tracker file"""

    @mock.patch("threading.enumerate")
    def test_async_save_tracker_file(self, mock_threading_enumerate):
        """Test async save tracker file."""
        async_save_manager = AsyncSaveManager(async_save='thread')
        tracker_filename = "./latest_checkpointed_iteration.txt"

        mock_threading_enumerate.side_effect = [
            # Simulate one thread name 'asyn_save_ckpt' running in first call
            [threading.Thread(name="asyn_save_ckpt")],
            # Simulate no threads running in second call
            [],
            # Simulate one thread name 'asyn_save_ckpt' running in third call
            [threading.Thread(name="asyn_save_ckpt")],
            # Simulate no threads running in fourth call
            [],
        ]

        # Simulate the first call to save_checkpoint
        self.async_save_manager_prepare(async_save_manager, tracker_filename, 1)
        # First call to maybe_finalize should not save the tracker file
        async_save_manager.maybe_finalize(wait_finish=False)
        self.assertFalse(os.path.exists(tracker_filename))
        # Second call to maybe_finalize should save the tracker file
        async_save_manager.maybe_finalize(wait_finish=False)
        self.assertTrue(os.path.exists(tracker_filename))
        with open(tracker_filename, "r") as f:
            content = f.read()
            self.assertEqual(content, "1")

        # Simulate the second call to save_checkpoint
        self.async_save_manager_prepare(async_save_manager, tracker_filename, 2)
        # Third call to maybe_finalize should not update the tracker file
        async_save_manager.maybe_finalize(wait_finish=False)
        with open(tracker_filename, "r") as f:
            content = f.read()
            self.assertEqual(content, "1")
        # Fourth call to maybe_finalize should update the tracker file
        async_save_manager.maybe_finalize(wait_finish=False)
        with open(tracker_filename, "r") as f:
            content = f.read()
            self.assertEqual(content, "2")
        # Clean up the tracker file
        os.remove(tracker_filename)

    def async_save_manager_prepare(self, async_save_manager, tracker_filename, iteration):
        """Prepare the async save manager with finalize function."""
        def iter_finalize_func():
            """Save tracker file."""
            with open(tracker_filename, "w") as f:
                f.write(str(iteration))

        async_save_manager.prepare_before_save()
        async_save_manager.add_finalize_fn(iter_finalize_func)
