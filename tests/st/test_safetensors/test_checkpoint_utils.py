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
"""test for load_checkpoint_utils."""
# pylint: disable=W0621
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
from mindspore import Parameter

from mindformers.tools.register import MindFormerConfig

from mindformers.checkpoint.utils import compile_model, check_checkpoints_dir_max_num, get_checkpoint_iter_dir, \
    get_checkpoint_tracker_filename, get_common_filename, get_metadata_filename, \
    get_latest_iteration_from_tracker, get_checkpoint_name, get_sharded_tensor_shard_id, \
    sharded_tensor_shard_id, _reverse_sharded_tensor_shard_id, _get_shard_size, verify_ckpt_valid, FileType
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.utils.load_checkpoint_utils import (
    CkptFormat, _get_checkpoint_mode, CheckpointFileMode, _check_checkpoint_path,
    extract_suffix, get_last_checkpoint, validate_config_with_file_mode,
    update_global_step, unify_safetensors, _revise_remove_redundancy_with_file,
    _get_origin_network, get_load_path_after_hf_convert, _get_src_strategy,
    _get_src_file_suffix, _get_src_file, load_safetensors_checkpoint,
    process_hf_checkpoint, validate_qkv_concat, get_merged_src_strategy_path,
    get_merged_dst_strategy_path, process_for_stand_alone_mode,
    load_checkpoint_with_safetensors
)


@pytest.fixture
def mock_config():
    """Create a mock config with default values"""

    class MockConfig:
        """Mock configuration class for testing"""

        def __init__(self):
            self.load_checkpoint = "/path/to/checkpoint"
            self.load_ckpt_format = "safetensors"
            self.use_parallel = False
            self.auto_trans_ckpt = False
            self.resume_training = None
            self.remove_redundancy = False
            self.output_dir = "/output"
            self.src_strategy_path_or_dir = None
            self.load_ckpt_async = False
            self.context = type('', (), {})()
            self.context.mode = "GRAPH_MODE"
            self.runner_config = type('', (), {})()
            self.runner_config.sink_mode = True
            self.runner_config.epochs = 1
            self.runner_config.sink_size = 1
            self.runner_config.step_scale = 2.0
            self.model = type('', (), {})()
            self.model.model_config = {}
            self.parallel = type('', (), {})()
            self.parallel.parallel_mode = "DATA_PARALLEL"

        def get(self, key, default=None):
            return getattr(self, key, default)

    return MockConfig()


@pytest.fixture
def mock_network():
    """Create a mock network"""
    mock_net = MagicMock()
    mock_net.cells.return_value = []
    return mock_net


@pytest.fixture
def mock_model():
    """Create a mock model"""
    mock_mod = MagicMock()
    mock_mod.config = MagicMock()
    mock_mod.config.model_type = "test_model"
    return mock_mod


@pytest.fixture
def mock_file():
    """Create a mock file"""
    mock_f = MagicMock()
    mock_f.metadata.return_value = None
    return mock_f


class TestCommonCheckpointMethod:
    """A test class for testing common methods"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_support_type(self):
        """test CkptFormat support type"""
        # run the test
        result = CkptFormat.support_type()

        # verify the results
        assert result == ['ckpt', 'safetensors']

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_checkpoint_path_with_non_string_pathlike(self):
        """test check checkpoint path with non string pathlike"""
        path = 123
        with pytest.raises(ValueError,
                           match=r"config.load_checkpoint must be a `str`, but got `123` as type `<class 'int'>`."):
            _check_checkpoint_path(path)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_checkpoint_path_with_nonexistent_path(self):
        """test check checkpoint path with nonexistent path"""
        path = 'NoneExistPath'
        with pytest.raises(FileNotFoundError, match=r"config.load_checkpoint `NoneExistPath` does not exist."):
            _check_checkpoint_path(path)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_checkpoint_path_with_valid_path(self):
        """test check checkpoint path with valid path"""
        # create a temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            # test with directory path
            result = _check_checkpoint_path(tmpdir)
            assert result == tmpdir

            # test with directory path ending with slash
            result = _check_checkpoint_path(tmpdir + '/')
            assert result == tmpdir

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        "file_path, expected",
        [
            # test pattern 1: {prefix}_rank_{rank_id}-{epoch}_{step}.safetensors
            ("model_rank_0-10_200.safetensors", "-10_200"),
            # test pattern 2: {prefix}_rank_{rank_id}_{task_id}-{epoch}_{step}.safetensors
            ("model_rank_0_1-10_200.safetensors", "_1-10_200"),
            # test with invalid pattern
            ("invalid_filename.safetensors", "invalid_filename")
        ]
    )
    def test_extract_suffix(self, file_path, expected):
        """test extract_suffix function"""
        result = extract_suffix(file_path)
        assert result == expected

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_last_checkpoint(self):
        """test get_last_checkpoint function"""
        # setup mocks using context managers
        with patch('os.path.isdir') as mock_isdir, \
                patch('os.path.exists') as mock_exists, \
                patch('os.listdir') as mock_listdir, \
                patch('os.path.getmtime') as mock_getmtime:
            # setup mock return values
            mock_isdir.return_value = True
            mock_exists.return_value = True
            mock_listdir.return_value = ["model_0.ckpt", "model_1.ckpt", "model_2.ckpt"]
            mock_getmtime.side_effect = lambda x: {
                "/test/model_0.ckpt": 100,
                "/test/model_1.ckpt": 200,
                "/test/model_2.ckpt": 300
            }[x]

            # test with valid directory
            result = get_last_checkpoint("/test", "ckpt")
            assert result == "/test/model_2.ckpt"

            # test with no checkpoint files
            mock_listdir.return_value = ["other_file.txt"]
            result = get_last_checkpoint("/test", "ckpt")
            assert result is None

            # test with invalid directory
            mock_isdir.return_value = False
            with pytest.raises(NotADirectoryError):
                get_last_checkpoint("/invalid/dir", "ckpt")

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        "file_mode, use_parallel, auto_trans_ckpt, expected_exception",
        [
            # test single checkpoint file mode with parallel
            (CheckpointFileMode.SINGLE_CHECKPOINT_FILE.value, True, False, ValueError),
            # test multi checkpoint file mode with parallel but no auto_trans_ckpt
            (CheckpointFileMode.MULTI_CHECKPOINT_FILE.value, True, False, ValueError),
            # test multi checkpoint file with rank id mode without parallel
            (CheckpointFileMode.MULTI_CHECKPOINT_FILE_WITH_RANK_ID.value, False, False, ValueError),
            # test invalid mode
            ("invalid_mode", False, False, ValueError),
            # test valid cases - no exception expected
            (CheckpointFileMode.SINGLE_CHECKPOINT_FILE.value, False, False, None),
            (CheckpointFileMode.MULTI_CHECKPOINT_FILE.value, True, True, None),
            (CheckpointFileMode.MULTI_CHECKPOINT_FILE_WITH_RANK_ID.value, True, False, None)
        ]
    )
    def test_validate_config_with_file_mode(self, file_mode, use_parallel, auto_trans_ckpt, expected_exception):
        """test validate_config_with_file_mode function"""
        if expected_exception:
            with pytest.raises(expected_exception):
                validate_config_with_file_mode(file_mode, use_parallel, auto_trans_ckpt)
        else:
            validate_config_with_file_mode(file_mode, use_parallel, auto_trans_ckpt)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        "step_scale, initial_global_step, expected_global_step, expected_in_dict",
        [
            (2.0, 100, 200, True),
            (None, 100, 100, True),
            (2.0, None, None, False)
        ]
    )
    def test_update_global_step(self, step_scale, initial_global_step, expected_global_step, expected_in_dict):
        """test update_global_step function"""
        # setup config
        config = type('', (), {})()
        config.runner_config = type('', (), {})()
        config.runner_config.step_scale = step_scale

        # setup hyper_param_dict
        hyper_param_dict = {}
        if initial_global_step is not None:
            hyper_param_dict["global_step"] = Parameter(np.array(initial_global_step, dtype=np.int32))

        # test update_global_step
        update_global_step(config, hyper_param_dict)

        # verify the results
        if expected_in_dict:
            assert "global_step" in hyper_param_dict
            assert hyper_param_dict["global_step"].asnumpy() == expected_global_step
        else:
            assert "global_step" not in hyper_param_dict

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_unify_safetensors(self):
        """test unify_safetensors function"""
        # setup mocks using context managers
        with patch('mindformers.utils.load_checkpoint_utils.is_main_rank') as mock_is_main_rank, \
                patch('mindformers.utils.load_checkpoint_utils.barrier') as mock_barrier, \
                patch('mindspore.unified_safetensors') as mock_unified_safetensors:
            # test when is_main_rank is True
            mock_is_main_rank.return_value = True
            unify_safetensors("/src/checkpoint", "/src/strategy", "/dst/unified", True, "-10_200", False)
            mock_unified_safetensors.assert_called_once()
            mock_barrier.assert_called_once()

            # test when is_main_rank is False
            mock_is_main_rank.return_value = False
            mock_barrier.reset_mock()
            unify_safetensors("/src/checkpoint", "/src/strategy", "/dst/unified", True, "-10_200", False)
            mock_unified_safetensors.assert_called_once()  # should not be called again
            mock_barrier.assert_called_once()

            # test without parallel
            mock_is_main_rank.return_value = True
            mock_barrier.reset_mock()
            unify_safetensors("/src/checkpoint", "/src/strategy", "/dst/unified", False, "-10_200", False)
            mock_barrier.assert_not_called()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        "config_remove_redundancy, metadata, expected_result",
        [
            # test with metadata remove_redundancy=True and config remove_redundancy=False
            (False, {"remove_redundancy": "True"}, True),
            # test with metadata remove_redundancy=False and config remove_redundancy=True
            (True, {"remove_redundancy": "False"}, False),
            # test with matching metadata and config
            (True, {"remove_redundancy": "True"}, True),
            # test with no metadata
            (True, None, True),
            # test with metadata but no remove_redundancy key
            (True, {"other_key": "value"}, True)
        ]
    )
    def test__revise_remove_redundancy_with_file(self, config_remove_redundancy, metadata, expected_result, mock_file):
        """test _revise_remove_redundancy_with_file function"""
        mock_file.metadata.return_value = metadata
        result = _revise_remove_redundancy_with_file(config_remove_redundancy, mock_file)
        assert result == expected_result

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        "network_has_convert_name, child_has_convert_name, expected_found",
        [
            # test with network that has convert_name
            (True, False, True),
            # test with nested network where child has convert_name
            (False, True, True),
            # test with network that doesn't have convert_name and no children with it
            (False, False, False)
        ]
    )
    def test__get_origin_network(self, network_has_convert_name, child_has_convert_name, expected_found):
        """test _get_origin_network function"""
        # setup mocks using context managers
        with patch('mindformers.utils.load_checkpoint_utils.logger'):
            if network_has_convert_name:
                # create a mock network with convert_name attribute
                mock_network = MagicMock()
                mock_network.convert_name = MagicMock()
                # Return empty list for cells() to avoid recursion
                mock_network.cells.return_value = []
            else:
                if child_has_convert_name:
                    # create a mock network without convert_name but with a child that has it
                    mock_child = MagicMock()
                    mock_child.convert_name = MagicMock()
                    # Return empty list for cells() to avoid further recursion
                    mock_child.cells.return_value = []

                    # Create a network that returns the child directly when cells() is called
                    mock_network = MagicMock()
                    mock_network.cells.return_value = [mock_child]
                else:
                    # create a mock network without convert_name and no children with it
                    mock_network = MagicMock()
                    mock_network.cells.return_value = []

                # Remove convert_name attribute to simulate network without it
                if hasattr(mock_network, 'convert_name'):
                    delattr(mock_network, 'convert_name')

            # run the test
            _, found = _get_origin_network(mock_network)
            assert found == expected_found

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_load_path_after_hf_convert(self, mock_config, mock_network):
        """test get_load_path_after_hf_convert function"""
        # setup mocks using context managers
        with patch('mindformers.utils.load_checkpoint_utils.is_hf_safetensors_dir') as mock_is_hf_safetensors_dir, \
                patch('mindformers.utils.load_checkpoint_utils.'
                      'check_safetensors_addition_param_support') as mock_check_support:
            # test when not hf safetensors
            mock_is_hf_safetensors_dir.return_value = False
            result = get_load_path_after_hf_convert(mock_config, mock_network)
            assert result == "/path/to/checkpoint"

            # test when hf safetensors but not qkv_concat and not supported
            mock_is_hf_safetensors_dir.return_value = True
            mock_check_support.return_value = False
            mock_config.model.model_config = {"qkv_concat": False}

            with patch('mindformers.utils.load_checkpoint_utils.process_hf_checkpoint',
                       return_value="/path/to/converted"):
                with patch('mindformers.utils.load_checkpoint_utils.barrier'):
                    result = get_load_path_after_hf_convert(mock_config, mock_network)
                    assert result == "/path/to/converted"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test__get_src_strategy(self, mock_config):
        """test _get_src_strategy function"""
        # setup mocks using context managers
        with patch('os.path.isfile') as mock_isfile, \
                patch('os.path.isdir') as mock_isdir, \
                patch('os.path.join') as mock_join, \
                patch('os.path.exists') as mock_exists, \
                patch('os.path.dirname') as mock_dirname, \
                patch('mindformers.utils.load_checkpoint_utils.logger'):
            # Test case 1: input_src_strategy is provided
            mock_config.load_checkpoint = "/test/checkpoint.ckpt"
            mock_config.src_strategy_path_or_dir = "/input/strategy"
            mock_isdir.return_value = True
            result = _get_src_strategy(mock_config)
            assert result == "/input/strategy"

            # Test case 2: no strategy dir exists
            mock_config.src_strategy_path_or_dir = None
            mock_isfile.return_value = True
            mock_exists.return_value = False

            with pytest.raises(
                    ValueError,
                    match="when use checkpoint after train/finetune, src_strategy_path_or_dir should be set"
            ):
                _get_src_strategy(mock_config)

            # Test case 3: config.load_checkpoint is a directory and strategy dir exists
            mock_isfile.return_value = False
            mock_exists.return_value = True

            # Setup mock_dirname to return a valid parent directory
            mock_dirname.return_value = "/test"

            # Setup mock_join to return a valid path
            mock_join.return_value = "/test/strategy"

            mock_config.load_checkpoint = "/test/checkpoint_dir"
            mock_config.src_strategy_path_or_dir = None

            result = _get_src_strategy(mock_config)
            assert result == "/test/strategy"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test__get_src_file_suffix(self, mock_config):
        """test _get_src_file_suffix function"""
        # setup mocks using context managers
        with patch('mindformers.utils.load_checkpoint_utils.is_main_rank') as mock_is_main_rank, \
                patch('mindformers.utils.load_checkpoint_utils.get_last_checkpoint') as mock_get_last_checkpoint, \
                patch('os.path.isfile') as mock_isfile, \
                patch('os.path.isdir') as mock_isdir:
            # test when is_main_rank is True and resume_training is string
            mock_is_main_rank.return_value = True
            mock_config.resume_training = "checkpoint-10_200.safetensors"
            mock_config.load_checkpoint = "/path/to/checkpoint"
            mock_config.load_ckpt_format = "safetensors"

            with patch('mindformers.utils.load_checkpoint_utils.extract_suffix', return_value="-10_200"):
                result = _get_src_file_suffix(mock_config)
                assert result == ("/path/to/checkpoint", "-10_200")

            # test when is_main_rank is True and load_checkpoint is file
            mock_isfile.return_value = True
            mock_isdir.return_value = False
            mock_config.resume_training = None
            mock_config.load_checkpoint = "/path/to/rank_0/checkpoint-10_200.safetensors"

            with patch('mindformers.utils.load_checkpoint_utils.extract_suffix', return_value="-10_200"):
                result = _get_src_file_suffix(mock_config)
                assert result == ("/path/to", "-10_200")

            # test when is_main_rank is True and load_checkpoint is dir
            mock_isfile.return_value = False
            mock_isdir.return_value = True
            mock_config.load_checkpoint = "/path/to/checkpoint"
            mock_get_last_checkpoint.return_value = "/path/to/checkpoint/rank_0/checkpoint-10_200.safetensors"

            with patch('mindformers.utils.load_checkpoint_utils.extract_suffix', return_value="-10_200"):
                result = _get_src_file_suffix(mock_config)
                assert result == ("/path/to/checkpoint", "-10_200")

            # test when is_main_rank is False
            mock_is_main_rank.return_value = False
            result = _get_src_file_suffix(mock_config)
            assert result == (None, None)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('mindformers.utils.load_checkpoint_utils.logger')
    def test__get_src_file(self, mock_logger):
        """test _get_src_file function"""
        # setup mocks using context managers
        with patch('os.path.exists') as mock_exists, \
                patch('os.path.join') as mock_join, \
                patch('mindformers.utils.load_checkpoint_utils.get_real_rank') as mock_get_real_rank, \
                patch('mindformers.utils.load_checkpoint_utils.get_last_checkpoint') as mock_get_last_checkpoint:
            # test with checkpoint_name provided
            mock_get_real_rank.return_value = 0
            mock_join.return_value = "/test/rank_0/checkpoint.ckpt"
            mock_exists.return_value = True

            result = _get_src_file("/test", "checkpoint.ckpt", "ckpt")
            assert result == "/test/rank_0/checkpoint.ckpt"

            # test without checkpoint_name
            mock_get_last_checkpoint.return_value = "/test/rank_0/last_checkpoint.ckpt"
            result = _get_src_file("/test", None, "ckpt")
            assert result == "/test/rank_0/last_checkpoint.ckpt"

            # test with non-existent file
            mock_exists.return_value = False
            with pytest.raises(FileNotFoundError):
                _get_src_file("/test", "non_existent.ckpt", "ckpt")

            # Verify that logger.error has been called.
            mock_logger.error.assert_called()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_safetensors_checkpoint(self, mock_config, mock_network, mock_file):
        """test load_safetensors_checkpoint function"""
        # Setup mocks using context managers
        with patch('mindformers.utils.load_checkpoint_utils._get_origin_network') as mock_get_origin_network, \
                patch('mindformers.utils.load_checkpoint_utils.ms') as mock_ms, \
                patch('mindformers.utils.load_checkpoint_utils.logger'), \
                patch('mindformers.utils.load_checkpoint_utils.safe_open') as mock_safe_open, \
                patch('mindformers.utils.load_checkpoint_utils.is_hf_safetensors_dir') as mock_is_hf_safetensors_dir:
            # Setup mock return values
            mock_get_origin_network.return_value = (MagicMock(), False)
            mock_ms.load_checkpoint.return_value = {"param1": MagicMock()}
            mock_is_hf_safetensors_dir.return_value = False

            # Mock the safe_open context manager
            mock_safe_open.return_value.__enter__.return_value = mock_file

            strategy_path = "/path/to/strategy"
            load_ckpt_path = "/path/to/checkpoint"
            optimizer = None

            load_safetensors_checkpoint(mock_config, ["/path/to/checkpoint.safetensors"], mock_network, strategy_path,
                                        load_ckpt_path,
                                        optimizer)
            mock_ms.load_param_into_net.assert_called_once()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_process_hf_checkpoint(self, mock_model, tmp_path):
        """test process_hf_checkpoint function"""
        # setup mocks using context managers
        with patch('mindformers.utils.load_checkpoint_utils.is_main_rank') as mock_is_main_rank, \
                patch('mindformers.utils.load_checkpoint_utils.barrier_world') as mock_barrier_world, \
                patch('mindformers.utils.load_checkpoint_utils.Process') as mock_process:
            # test when is_main_rank is True
            mock_is_main_rank.return_value = True
            mock_process_instance = MagicMock()
            mock_process_instance.exitcode = 0
            mock_process.return_value = mock_process_instance

            # Use tmp_path for output and input paths
            output_dir = tmp_path / "output" / "dir"
            input_checkpoint = tmp_path / "input" / "checkpoint"
            # Create input directory
            input_checkpoint.parent.mkdir(parents=True, exist_ok=True)

            result = process_hf_checkpoint(mock_model, str(output_dir), str(input_checkpoint))
            expected_path = str(output_dir / "test_model_ms_converted_weight")
            assert result == expected_path
            mock_process_instance.start.assert_called_once()
            mock_process_instance.join.assert_called_once()
            mock_barrier_world.assert_called_once()

            # Reset mocks for next test case
            mock_process.reset_mock()
            mock_process_instance = MagicMock()
            mock_process_instance.exitcode = 1
            mock_process.return_value = mock_process_instance

            # test when process exits with error
            with pytest.raises(RuntimeError, match="convert HuggingFace weight failed."):
                process_hf_checkpoint(mock_model, str(output_dir), str(input_checkpoint))

            # Reset mocks for next test case
            mock_process.reset_mock()
            mock_process_instance = MagicMock()
            mock_process_instance.exitcode = 0
            mock_process.return_value = mock_process_instance

            # test when is_main_rank is False
            mock_is_main_rank.return_value = False
            process_hf_checkpoint(mock_model, str(output_dir), str(input_checkpoint))
            mock_process_instance.start.assert_not_called()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        "model, qkv_concat_config, check_safetensors_key_return, "
        "has_concat_keys, expected_exception, should_log_warning",
        [
            # test with non-PreTrainedModel
            ("not_a_model", False, False, False, None, True),
            # test with PreTrainedModel but no concat keys
            (MagicMock(spec=PreTrainedModel), False, False, False, None, False),
            # Test case where check_safetensors_key returns True and qkv_concat_config is True
            (MagicMock(spec=PreTrainedModel), True, True, True, None, False),
            # Test case where check_safetensors_key returns False and qkv_concat_config is True
            (MagicMock(spec=PreTrainedModel), True, False, True, ValueError, False),
            # Test case where check_safetensors_key returns True and qkv_concat_config is False
            (MagicMock(spec=PreTrainedModel), False, True, True, ValueError, False)
        ]
    )
    def test_validate_qkv_concat(self, model, qkv_concat_config,
                                 check_safetensors_key_return, has_concat_keys, expected_exception, should_log_warning):
        """test validate_qkv_concat function"""
        # Setup mocks using context managers
        with patch('mindformers.utils.load_checkpoint_utils.logger') as mock_logger, \
                patch('mindformers.utils.load_checkpoint_utils.check_safetensors_key') as mock_check_safetensors_key:

            # Setup mock behavior
            mock_check_safetensors_key.return_value = check_safetensors_key_return

            # If it's a PreTrainedModel, set up obtain_qkv_ffn_concat_keys
            if hasattr(model, 'obtain_qkv_ffn_concat_keys'):
                model.obtain_qkv_ffn_concat_keys.return_value = ["qkv_concat_key"] if has_concat_keys else None

            # Run the test and check results
            if expected_exception:
                with pytest.raises(expected_exception, match="The qkv concat check failed!"):
                    validate_qkv_concat(model, qkv_concat_config, "/path/to/checkpoint")
            else:
                validate_qkv_concat(model, qkv_concat_config, "/path/to/checkpoint")

            # Check if warning was logged when expected
            if should_log_warning:
                mock_logger.warning.assert_called_once()
            else:
                mock_logger.warning.assert_not_called()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_merged_src_strategy_path(self, mock_config):
        """test get_merged_src_strategy_path function"""
        # setup mocks using context managers
        with patch('mindformers.utils.load_checkpoint_utils.is_main_rank') as mock_is_main_rank, \
                patch('mindformers.utils.load_checkpoint_utils.barrier') as mock_barrier, \
                patch('mindformers.utils.load_checkpoint_utils._get_src_strategy') as mock_get_src_strategy, \
                patch('mindformers.utils.load_checkpoint_utils.ms.merge_pipeline_strategys') as mock_merge_strategys, \
                patch('os.makedirs'):
            # test when is_main_rank is True
            mock_is_main_rank.return_value = True
            mock_get_src_strategy.return_value = "/input/strategy"

            result = get_merged_src_strategy_path(mock_config)
            assert result == "/output/merged_strategy/src_strategy.ckpt"
            mock_merge_strategys.assert_called_once()
            mock_barrier.assert_called_once()

            # test when is_main_rank is False
            mock_is_main_rank.return_value = False
            mock_barrier.reset_mock()
            result = get_merged_src_strategy_path(mock_config)
            mock_merge_strategys.assert_called_once()  # should not be called again
            mock_barrier.assert_called_once()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_merged_dst_strategy_path(self, mock_config):
        """test get_merged_dst_strategy_path function"""
        # setup mocks using context managers
        with patch('mindformers.utils.load_checkpoint_utils.is_main_rank') as mock_is_main_rank, \
                patch('mindformers.utils.load_checkpoint_utils.barrier') as mock_barrier, \
                patch('mindformers.utils.load_checkpoint_utils.ms.merge_pipeline_strategys') as mock_merge_strategys, \
                patch('os.makedirs'):
            # test with use_parallel=True, auto_trans_ckpt=True, not stand_alone
            mock_is_main_rank.return_value = True

            mock_config.use_parallel = True
            mock_config.auto_trans_ckpt = True
            mock_config.parallel.parallel_mode = "DATA_PARALLEL"

            strategy_path = "/path/to/strategy.ckpt"

            result = get_merged_dst_strategy_path(mock_config, strategy_path)
            assert result == "/output/merged_strategy/dst_strategy.ckpt"
            mock_merge_strategys.assert_called_once()
            mock_barrier.assert_called_once()

            # test with stand_alone mode
            mock_config.parallel.parallel_mode = "STAND_ALONE"
            result = get_merged_dst_strategy_path(mock_config, strategy_path)
            assert result == "/path/to/strategy.ckpt"

            # test with use_parallel=False
            mock_config.use_parallel = False
            result = get_merged_dst_strategy_path(mock_config, strategy_path)
            assert result == "/path/to/strategy.ckpt"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_process_for_stand_alone_mode(self, mock_config, mock_network):
        """test process_for_stand_alone_mode function"""
        strategy_path = "/path/to/strategy.ckpt"

        # setup mocks using context managers
        with patch('mindformers.utils.load_checkpoint_utils._pynative_executor'), \
                patch('mindformers.utils.load_checkpoint_utils.is_main_rank') as mock_is_main_rank, \
                patch('mindformers.utils.load_checkpoint_utils.barrier') as mock_barrier, \
                patch('mindformers.utils.load_checkpoint_utils.generate_state_dict') as mock_generate_state_dict, \
                patch('mindformers.utils.load_checkpoint_utils.save_strategy_file') as mock_save_strategy_file, \
                patch('os.makedirs') as mock_makedirs, \
                patch('shutil.rmtree') as mock_rmtree, \
                patch('os.path.exists') as mock_exists:
            # test with stand_alone mode
            mock_is_main_rank.return_value = True
            mock_exists.return_value = True
            mock_config.parallel.parallel_mode = "STAND_ALONE"
            mock_config.use_parallel = True

            process_for_stand_alone_mode(mock_config, mock_network, strategy_path)
            mock_rmtree.assert_called_once()
            mock_makedirs.assert_called_once()
            mock_generate_state_dict.assert_called_once()
            mock_save_strategy_file.assert_called_once()
            mock_barrier.assert_called()

            # Reset mocks for next test case
            mock_barrier.reset_mock()
            mock_rmtree.reset_mock()
            mock_makedirs.reset_mock()
            mock_generate_state_dict.reset_mock()
            mock_save_strategy_file.reset_mock()

            # test when strategy dir doesn't exist
            mock_exists.return_value = False
            process_for_stand_alone_mode(mock_config, mock_network, strategy_path)
            mock_rmtree.assert_not_called()

            # Reset mocks for next test case
            mock_barrier.reset_mock()
            mock_rmtree.reset_mock()
            mock_makedirs.reset_mock()
            mock_generate_state_dict.reset_mock()
            mock_save_strategy_file.reset_mock()

            # test when not stand_alone mode
            mock_config.parallel.parallel_mode = "DATA_PARALLEL"
            process_for_stand_alone_mode(mock_config, mock_network, strategy_path)
            mock_rmtree.assert_not_called()
            mock_barrier.assert_not_called()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_load_checkpoint_with_safetensors(self, mock_config, mock_model, mock_network):
        """test load_checkpoint_with_safetensors function"""
        # setup mocks using context managers
        with patch('mindformers.utils.load_checkpoint_utils._check_checkpoint_path') as mock_check_checkpoint_path, \
                patch('mindformers.utils.load_checkpoint_utils._get_checkpoint_mode') as mock_get_checkpoint_mode, \
                patch('mindformers.utils.load_checkpoint_utils.'
                      'validate_config_with_file_mode') as mock_validate_config_with_file_mode, \
                patch('mindformers.utils.load_checkpoint_utils.compile_model') as mock_compile_model, \
                patch('mindformers.utils.load_checkpoint_utils.validate_qkv_concat'), \
                patch('mindformers.utils.load_checkpoint_utils.process_for_stand_alone_mode'), \
                patch('mindformers.utils.load_checkpoint_utils.'
                      'get_merged_dst_strategy_path') as mock_get_merged_dst_strategy_path, \
                patch('mindformers.utils.load_checkpoint_utils.'
                      'load_safetensors_checkpoint') as mock_load_safetensors_checkpoint, \
                patch('mindformers.utils.load_checkpoint_utils.logger'), \
                patch('mindformers.utils.load_checkpoint_utils.barrier'):
            # setup mocks return values
            mock_check_checkpoint_path.return_value = "/valid/checkpoint"
            mock_get_checkpoint_mode.return_value = CheckpointFileMode.SINGLE_CHECKPOINT_FILE.value
            mock_get_merged_dst_strategy_path.return_value = "/path/to/merged/strategy"

            # setup input_data and optimizer
            input_data = MagicMock()
            optimizer = None

            # test with do_eval=True
            load_checkpoint_with_safetensors(mock_config, mock_model, mock_network, input_data, do_eval=True,
                                             do_predict=False,
                                             optimizer=optimizer)
            mock_check_checkpoint_path.assert_called_once()
            mock_get_checkpoint_mode.assert_called_once()
            mock_validate_config_with_file_mode.assert_called_once()
            mock_load_safetensors_checkpoint.assert_called_once()

            # test with do_predict=True
            mock_load_safetensors_checkpoint.reset_mock()
            load_checkpoint_with_safetensors(mock_config, mock_model, mock_network, input_data, do_eval=False,
                                             do_predict=True,
                                             optimizer=optimizer)
            mock_load_safetensors_checkpoint.assert_called_once()

            # test with use_parallel=True
            mock_config.use_parallel = True
            mock_load_safetensors_checkpoint.reset_mock()
            mock_compile_model.reset_mock()
            load_checkpoint_with_safetensors(mock_config, mock_model, mock_network, input_data, do_eval=False,
                                             do_predict=False,
                                             optimizer=optimizer)
            mock_compile_model.assert_called_once()
            mock_load_safetensors_checkpoint.assert_called_once()

            # test with resume_training=True
            mock_config.resume_training = True
            # Access protected member for testing purposes
            # pylint: disable=W0212
            mock_model._train_network = MagicMock()
            mock_load_safetensors_checkpoint.reset_mock()
            load_checkpoint_with_safetensors(mock_config, mock_model, mock_network, input_data, do_eval=False,
                                             do_predict=False,
                                             optimizer=optimizer)
            mock_load_safetensors_checkpoint.assert_called_once()


class TestBuildModel:
    """A test class for testing build_model"""
    runner_config = {'sink_mode': True, 'epochs': 1, 'sink_size': 1}
    config = {
        'runner_config': runner_config,
        'context': {'mode': 0}  # 0 is typically ms.GRAPH_MODE, 1 is ms.PYNATIVE_MODE
    }
    model = MagicMock()
    dataset = MagicMock()

    @patch('mindspore.context.get_auto_parallel_context')
    def test_build_model_sink_mode_value_error(self, mock_get_auto_parallel_context):
        """test build model sink mode value error"""
        mock_get_auto_parallel_context.return_value = 'auto_parallel'
        config = MindFormerConfig(**self.config)
        config.runner_config.sink_mode = False
        with pytest.raises(ValueError):
            compile_model(
                model=None,
                dataset=None,
                mode=config.context.mode,
                sink_mode=config.runner_config.sink_mode,
                epoch=config.runner_config.epochs,
                sink_size=config.runner_config.sink_size,
                do_eval=False, do_predict=False
            )

    @patch('mindspore.context.get_auto_parallel_context')
    def test_build_model_infer_predict_layout_when_do_eval_is_true(self, mock_get_auto_parallel_context):
        """test build model infer predict layout when do eval is true"""
        mock_get_auto_parallel_context.return_value = 'auto_parallel'
        config = MindFormerConfig(**self.config)
        compile_model(
            model=self.model,
            dataset=self.dataset,
            mode=config.context.mode,
            sink_mode=config.runner_config.sink_mode,
            epoch=config.runner_config.epochs,
            sink_size=config.runner_config.sink_size,
            do_eval=True, do_predict=False
        )
        self.model.infer_predict_layout.assert_called_once_with(*self.dataset)

    @patch('mindspore.context.get_auto_parallel_context')
    def test_build_model_infer_predict_layout_when_do_predict_is_true(self, mock_get_auto_parallel_context):
        """test build model infer predict layout when do predict is true"""
        mock_get_auto_parallel_context.return_value = 'auto_parallel'
        config = MindFormerConfig(**self.config)
        compile_model(
            model=self.model,
            dataset=self.dataset,
            mode=config.context.mode,
            sink_mode=config.runner_config.sink_mode,
            epoch=config.runner_config.epochs,
            sink_size=config.runner_config.sink_size,
            do_eval=False, do_predict=True
        )
        self.model.infer_predict_layout.assert_called_once_with(*self.dataset)

    @patch('mindspore.context.get_auto_parallel_context')
    def test_build_model_model_build(self, mock_get_auto_parallel_context):
        """test build model build"""
        mock_get_auto_parallel_context.return_value = 'auto_parallel'
        config = MindFormerConfig(**self.config)
        compile_model(
            model=self.model,
            dataset=self.dataset,
            mode=config.context.mode,
            sink_mode=config.runner_config.sink_mode,
            epoch=config.runner_config.epochs,
            sink_size=config.runner_config.sink_size,
            do_eval=False, do_predict=False
        )
        self.model.build.assert_called_once_with(train_dataset=self.dataset, epoch=1, sink_size=1)


class TestGetCheckpointMode:
    """A test class for testing get_checkpoint_mode"""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch('os.path.isfile')
    @patch('os.path.isdir')
    def test_single_checkpoint_file(self, mock_isdir, mock_isfile):
        """test single checkpoint file"""
        mock_isfile.return_value = True
        mock_isdir.return_value = False
        config = type('', (), {})()
        config.load_checkpoint = '/test/checkpoint_file.safetensors'
        assert _get_checkpoint_mode(config) == CheckpointFileMode.SINGLE_CHECKPOINT_FILE.value

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_multi_checkpoint_file_with_rank_id(self):
        """test multi checkpoint file with rank id"""
        with patch('os.path.isfile') as mock_isfile, \
                patch('os.path.isdir') as mock_isdir:
            mock_isfile.return_value = False
            mock_isdir.return_value = True
            with patch('os.listdir', return_value=['rank_0']):
                config = type('', (), {})()
                config.load_checkpoint = '/test/checkpoint_dir/'
                assert _get_checkpoint_mode(config) == CheckpointFileMode.MULTI_CHECKPOINT_FILE_WITH_RANK_ID.value

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_multi_checkpoint_file(self):
        """ test multi checkpoint file"""
        with patch('os.path.isfile') as mock_isfile, \
                patch('os.path.isdir') as mock_isdir:
            mock_isfile.return_value = False
            mock_isdir.return_value = True
            with patch('os.listdir', return_value=['checkpoint.safetensors']):
                config = type('', (), {})()
                config.load_checkpoint = '/test/checkpoint_dir/'
                config.load_ckpt_format = '.safetensors'
                assert _get_checkpoint_mode(config) == CheckpointFileMode.MULTI_CHECKPOINT_FILE.value

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_invalid_path(self):
        """test invalid path"""
        with patch('os.path.isfile') as mock_isfile, \
                patch('os.path.isdir') as mock_isdir:
            mock_isfile.return_value = False
            mock_isdir.return_value = False
            config = type('', (), {})()
            config.load_checkpoint = 'invalid_path'
            with pytest.raises(ValueError, match="Provided path is neither a file nor a directory."):
                _get_checkpoint_mode(config)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_no_valid_checkpoint_files(self):
        """test no valid checkpoint files"""
        with patch('os.path.isfile') as mock_isfile, \
                patch('os.path.isdir') as mock_isdir:
            mock_isfile.return_value = False
            mock_isdir.return_value = True
            with patch('os.listdir', return_value=['not_a_checkpoint_file']):
                config = type('', (), {})()
                config.load_checkpoint = '/test/checkpoint_dir/'
                config.load_ckpt_format = '.safetensors'
                with pytest.raises(ValueError, match="not support mode: no valid checkpoint files found"):
                    _get_checkpoint_mode(config)


class TestCheckpointUtils:
    """A test class for testing checkpoint utils functions"""

    # Test get_checkpoint_iter_dir function
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_checkpoint_iter_dir(self):
        """test get_checkpoint_iter_dir function"""
        checkpoints_path = '/test/checkpoints'
        iteration = 123
        result = get_checkpoint_iter_dir(checkpoints_path, iteration)
        # Use os.path.normpath to handle different path separators
        assert os.path.normpath(result) == os.path.normpath('/test/checkpoints/iteration_00000123')

        # Test with different iteration format
        iteration = 1000
        result = get_checkpoint_iter_dir(checkpoints_path, iteration)
        assert os.path.normpath(result) == os.path.normpath('/test/checkpoints/iteration_00001000')

    # Test get_checkpoint_tracker_filename function
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_checkpoint_tracker_filename(self):
        """test get_checkpoint_tracker_filename function"""
        checkpoints_path = '/test/checkpoints'
        result = get_checkpoint_tracker_filename(checkpoints_path)
        assert os.path.normpath(result) == os.path.normpath('/test/checkpoints/latest_checkpointed_iteration.txt')

    # Test get_common_filename function
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_common_filename(self):
        """test get_common_filename function"""
        checkpoints_path = '/test/checkpoints'
        iteration = 123
        result = get_common_filename(checkpoints_path, iteration)
        assert os.path.normpath(result) == os.path.normpath('/test/checkpoints/iteration_00000123/common.json')

    # Test get_metadata_filename function
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_metadata_filename(self):
        """test get_metadata_filename function"""
        checkpoints_path = '/test/checkpoints'
        iteration = 123
        result = get_metadata_filename(checkpoints_path, iteration)
        assert os.path.normpath(result) == os.path.normpath('/test/checkpoints/iteration_00000123/metadata.json')

    # Test get_checkpoint_name function
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_checkpoint_name(self):
        """test get_checkpoint_name function"""
        # Test with cur_iter_checkpoint_dir and user_prefix
        cur_iter_checkpoint_dir = '/test/checkpoints/iteration_00000123'
        user_prefix = 'model'
        file_idx = 0
        total_file_num = 1
        file_type = FileType.MODEL
        result = get_checkpoint_name(cur_iter_checkpoint_dir, user_prefix, file_idx, total_file_num, file_type)
        expected = '/test/checkpoints/iteration_00000123/model-model-0000000-0000001'
        assert os.path.normpath(result) == os.path.normpath(expected)

        # Test with optimizer type
        file_type = FileType.OPTIMIZER
        result = get_checkpoint_name(cur_iter_checkpoint_dir, user_prefix, file_idx, total_file_num, file_type)
        expected = '/test/checkpoints/iteration_00000123/model-opt-0000000-0000001'
        assert os.path.normpath(result) == os.path.normpath(expected)

        # Test without user_prefix
        result = get_checkpoint_name(cur_iter_checkpoint_dir, None, file_idx, total_file_num, FileType.MODEL)
        expected = '/test/checkpoints/iteration_00000123/model-0000000-0000001'
        assert os.path.normpath(result) == os.path.normpath(expected)

        # Test without cur_iter_checkpoint_dir
        result = get_checkpoint_name(None, None, file_idx, total_file_num, FileType.MODEL)
        expected = 'model-0000000-0000001'
        assert result == expected

    # Test sharded tensor related functions
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_sharded_tensor_shard_id_functions(self):
        """test sharded tensor shard id functions"""
        param_name = 'model.layer.weight'
        global_offset = (100, 200)

        # Test get_sharded_tensor_shard_id
        shard_id1 = get_sharded_tensor_shard_id(param_name, global_offset)
        expected = "('model.layer.weight', (100, 200))"
        assert shard_id1 == expected

        # Test sharded_tensor_shard_id (duplicate function)
        shard_id2 = sharded_tensor_shard_id(param_name, global_offset)
        assert shard_id2 == expected
        assert shard_id1 == shard_id2

    # Test _reverse_sharded_tensor_shard_id function
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_reverse_sharded_tensor_shard_id(self):
        """test _reverse_sharded_tensor_shard_id function"""
        # Test normal case
        shard_id = "('model.layer.weight', (100, 200))"
        param_name, global_offset = _reverse_sharded_tensor_shard_id(shard_id)
        assert param_name == 'model.layer.weight'
        assert global_offset == (100, 200)

        # Test with empty offset
        shard_id = "('model.layer.weight', ())"
        param_name, global_offset = _reverse_sharded_tensor_shard_id(shard_id)
        assert param_name == 'model.layer.weight'
        assert not global_offset

        # Test with single element offset
        shard_id = "('model.layer.weight', (50,))"
        param_name, global_offset = _reverse_sharded_tensor_shard_id(shard_id)
        assert param_name == 'model.layer.weight'
        assert global_offset == (50,)

        # Test invalid shard id
        with pytest.raises(ValueError):
            _reverse_sharded_tensor_shard_id("invalid_shard_id")

    # Test _get_shard_size function
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_shard_size(self):
        """test _get_shard_size function"""
        # Test with float32 type (32 bits per float32)
        local_shape = (100, 200)
        dtype = 'Float32'
        expected = 100 * 200 * 32  # 100*200 elements * 32 bits each
        assert _get_shard_size(local_shape, dtype) == expected

        # Test with int8 type (8 bits per int8)
        dtype = 'Int8'
        expected = 100 * 200 * 8  # 100*200 elements * 8 bits each
        assert _get_shard_size(local_shape, dtype) == expected

        # Test with unknown dtype (should default to 16 bits)
        dtype = 'UnknownType'
        expected = 100 * 200 * 16  # 100*200 elements * 16 bits default
        assert _get_shard_size(local_shape, dtype) == expected

        # Test with empty shape
        local_shape = ()
        dtype = 'Float32'
        expected = 1 * 32  # scalar, 32 bits
        assert _get_shard_size(local_shape, dtype) == expected

    # Test verify_ckpt_valid function with tmp_path
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_verify_ckpt_valid(self, tmp_path):
        """test verify_ckpt_valid function"""
        # Test with valid directory containing safetensors file
        ckpt_dir = tmp_path / "valid_ckpt"
        ckpt_dir.mkdir()
        safetensor_file = ckpt_dir / "model.safetensors"
        safetensor_file.touch()
        assert verify_ckpt_valid(str(ckpt_dir)) is None

        # Test with valid directory containing metadata and safetensors file
        ckpt_dir = tmp_path / "valid_ckpt_with_metadata"
        ckpt_dir.mkdir()
        metadata_path = ckpt_dir / "metadata.json"
        metadata_content = {
            "storage_data": {
                "param1": [{
                    "file_name": "model.safetensors"
                }]
            }
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_content, f)
        safetensor_file = ckpt_dir / "model.safetensors"
        safetensor_file.touch()
        assert verify_ckpt_valid(str(ckpt_dir)) is None

        # Test with invalid directory (not exists)
        with pytest.raises(NotADirectoryError):
            verify_ckpt_valid("/non/existent/directory")

        # Test with directory containing no files
        ckpt_dir = tmp_path / "empty_ckpt"
        ckpt_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            verify_ckpt_valid(str(ckpt_dir))

        # Test with metadata referencing missing safetensors file
        ckpt_dir = tmp_path / "invalid_metadata_ckpt"
        ckpt_dir.mkdir()
        metadata_path = ckpt_dir / "metadata.json"
        metadata_content = {
            "storage_data": {
                "param1": [{
                    "file_name": "missing.safetensors"
                }]
            }
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_content, f)
        with pytest.raises(FileNotFoundError):
            verify_ckpt_valid(str(ckpt_dir))

        # Test with invalid metadata json
        ckpt_dir = tmp_path / "invalid_json_ckpt"
        ckpt_dir.mkdir()
        metadata_path = ckpt_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("invalid json content")
        with pytest.raises(RuntimeError):
            verify_ckpt_valid(str(ckpt_dir))

    # Test check_checkpoints_dir_max_num function with tmp_path
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_check_checkpoints_dir_max_num(self, tmp_path):
        """test check_checkpoints_dir_max_num function"""
        # Create test directory structure
        checkpoints_root_path = tmp_path / "checkpoints"
        checkpoints_root_path.mkdir()

        # Create more directories than max_keep_num
        for i in range(5):
            dir_path = checkpoints_root_path / f"iteration_{i:08d}"
            dir_path.mkdir()

        # Test with max_keep_num = 3, should keep newest 3 directories
        check_checkpoints_dir_max_num(3, str(checkpoints_root_path))

        # Verify only 3 directories remain
        remaining_dirs = list(checkpoints_root_path.iterdir())
        remaining_dirs.sort()
        assert len(remaining_dirs) == 3
        assert [d.name for d in remaining_dirs] == ["iteration_00000002", "iteration_00000003", "iteration_00000004"]

        # Test with max_keep_num larger than existing directories
        check_checkpoints_dir_max_num(10, str(checkpoints_root_path))
        remaining_dirs = list(checkpoints_root_path.iterdir())
        assert len(remaining_dirs) == 3

    # Test get_latest_iteration_from_tracker function with tmp_path
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_latest_iteration_from_tracker(self, tmp_path):
        """test get_latest_iteration_from_tracker function"""
        checkpoints_path = tmp_path / "checkpoints"
        checkpoints_path.mkdir()

        # Create tracker file
        tracker_path = checkpoints_path / "latest_checkpointed_iteration.txt"
        tracker_path.write_text("123", encoding="utf-8")

        # Create corresponding directory
        iter_dir = checkpoints_path / "iteration_00000123"
        iter_dir.mkdir()

        # Test normal case
        assert get_latest_iteration_from_tracker(str(checkpoints_path)) == 123

        # Test with missing tracker file
        tracker_path.unlink()
        with pytest.raises(FileNotFoundError):
            get_latest_iteration_from_tracker(str(checkpoints_path))

        # Test with invalid iteration number in tracker file
        tracker_path.write_text("invalid_iter", encoding="utf-8")
        with pytest.raises(ValueError):
            get_latest_iteration_from_tracker(str(checkpoints_path))

        # Test with missing iteration directory
        tracker_path.write_text("456", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            get_latest_iteration_from_tracker(str(checkpoints_path))
