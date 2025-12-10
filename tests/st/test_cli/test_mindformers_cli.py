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
System tests for mindformers_cli.py

This module contains system tests for the mindformers CLI tool, including:
- Single card mode tests for training and inference
- Multi-card mode tests for training and inference (single node only)

These tests verify the CLI interface using real configuration files based on
llm_template_v2_experimental templates.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

import mindformers
from mindformers.tools.logger import logger
from .generate_fake_dataset import generate_fake_dataset
from .generate_fake_qwen3_model import create_fake_qwen3_model

# Get the project root directory (mindformers installation path)
PROJECT_ROOT = Path(mindformers.__file__).parent.parent

PROJECT_CONFIG_DIR = PROJECT_ROOT / "configs" / "llm_template_v2_experimental"
PRETRAIN_CONFIG_TEMPLATE = PROJECT_CONFIG_DIR / "llm_pretrain_template.yaml"
PREDICT_CONFIG_TEMPLATE = PROJECT_CONFIG_DIR / "llm_predict_template.yaml"


class TestMindFormersCLI:
    """System test class for mindformers_cli.py"""

    @pytest.fixture(scope="class", autouse=True)
    def setup_class_data(self, tmp_path_factory):
        """Setup test data once for all test methods in the class"""
        # Create a shared temporary directory for all tests
        base_tmp_path = tmp_path_factory.mktemp("cli_test_data")

        # Generate fake dataset and model directory (once for all tests)
        data_dir = base_tmp_path / "test_data"
        model_dir = base_tmp_path / "test_models" / "fake_qwen3"
        data_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Generate fake dataset
        dataset_prefix = data_dir / "test_megatron"
        logger.info("Generating fake dataset (shared for all tests)...")
        generate_fake_dataset(
            output_prefix=str(dataset_prefix),
            num_samples=100,
            seq_length=1024
        )
        dataset_path = f"{dataset_prefix}_text_document"

        # Generate fake model directory
        logger.info("Generating fake Qwen3 model directory (shared for all tests)...")
        create_fake_qwen3_model(
            model_dir=str(model_dir),
            vocab_size=1000,
            hidden_size=512
        )
        model_dir_str = str(model_dir)

        # Generate temporary parallel_speed_up.json file (once for all tests)
        parallel_speed_up_json_path = base_tmp_path / "parallel_speed_up.json"
        with open(parallel_speed_up_json_path, 'w', encoding='utf-8') as f:
            json.dump({"dataset_broadcast_opt_level": 3}, f, indent=4)

        # Store template config paths and test data paths
        # All config modifications will be done via CLI arguments, not by modifying YAML files
        TestMindFormersCLI.base_tmp_path = base_tmp_path
        TestMindFormersCLI.pretrain_config_template_path = PRETRAIN_CONFIG_TEMPLATE
        TestMindFormersCLI.predict_config_template_path = PREDICT_CONFIG_TEMPLATE
        TestMindFormersCLI.model_dir = model_dir_str
        TestMindFormersCLI.dataset_path = dataset_path
        TestMindFormersCLI.parallel_speed_up_json_path = str(parallel_speed_up_json_path)

    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        """Setup test environment for each test method"""
        self.tmp_path = tmp_path
        self.log_dir = tmp_path / "test_logs"
        self.log_dir.mkdir(exist_ok=True)

        # Use shared template config paths and test data from class setup
        self.pretrain_config_template_path = TestMindFormersCLI.pretrain_config_template_path
        self.predict_config_template_path = TestMindFormersCLI.predict_config_template_path
        self.model_dir = TestMindFormersCLI.model_dir
        self.dataset_path = TestMindFormersCLI.dataset_path
        self.parallel_speed_up_json_path = TestMindFormersCLI.parallel_speed_up_json_path

    def check_unexpected_errors(self, result):
        """
        Check for unexpected errors in command output that should cause test to fail.

        Args:
            result: subprocess.CompletedProcess result

        Raises:
            pytest.fail: If unexpected error is detected
        """
        if result.returncode == 0:
            return

        output = result.stdout + result.stderr
        error_lower = output.lower()

        # Comprehensive list of unexpected errors that should cause test to fail
        unexpected_errors = [
            # Command/system errors
            "command not found",
            "command not recognized",
            "is not recognized as an internal or external command",
            "no such file or directory",
            "cannot access",
            "permission denied",
            "access denied",
            "file not found",
            "directory not found",
            "path not found",

            # Python syntax/compilation errors
            "syntaxerror",
            "indentationerror",
            "taberror",
            "nameerror",
            "typeerror",
            "valueerror",
            "attributeerror",
            "keyerror",
            "indexerror",
            "zerodivisionerror",

            # Python import/module errors
            "importerror",
            "modulenotfounderror",
            "importlib",

            # Python runtime errors
            "runtimeerror",
            "notimplementederror",
            "recursionerror",
            "memoryerror",
            "overflowerror",
            "referenceerror",
            "systemerror",

            # File/IO errors
            "ioerror",
            "oserror",
            "filenotfounderror",
            "permissionerror",
            "isadirectoryerror",
            "notadirectoryerror",

            # Configuration/parsing errors
            "yaml.error",
            "yaml.scannererror",
            "yaml.parsererror",
            "json.decodererror",
            "configerror",
            "configuration error",
            "invalid configuration",
            "parse error",
            "parsing error",

            # Environment/setup errors
            "environment variable",
            "environment not set",
            "missing environment",
            "setup error",
            "initialization error",
            "initialization failed",

            # Argument/parameter errors
            "argument error",
            "invalid argument",
            "missing required argument",
            "unrecognized argument",
            "argumentparser",

            # Execution errors
            "execution error",
            "execution failed",
            "failed to execute",
            "execution aborted",
            "aborted",

            # Traceback/exception indicators (when not expected)
            "traceback (most recent call last)",
            "exception:",
            "error:",
            "fatal error",
            "critical error",
            "unhandled exception",
            "uncaught exception",

            # System/OS errors
            "system error",
            "os error",
            "kernel error",
            "segmentation fault",
            "bus error",
            "abort",
            "core dumped",
        ]

        for error in unexpected_errors:
            if error.lower() in error_lower:
                pytest.fail(
                    f"Command failed with unexpected error: {error}\n"
                    f"Return code: {result.returncode}\n"
                    f"Stdout: {result.stdout}\n"
                    f"Stderr: {result.stderr}"
                )

    def print_worker_log(self, log_dir):
        """
        Print worker_0.log content from msrun log directory.
        The log directory is set via --log-dir parameter in CLI.

        Args:
            log_dir: Log directory path (can be relative or absolute)
        """
        try:
            log_dir_path = Path(log_dir)

            # Construct log path: log_dir/worker_0.log
            log_path = log_dir_path / "worker_0.log"

            if log_path.exists():
                logger.info(f"Reading worker log from: {log_path}")
                try:
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        log_content = f.read()
                        if log_content:
                            logger.info("=" * 80)
                            logger.info(f"Worker 0 Log Content ({log_path}):")
                            logger.info("=" * 80)
                            logger.info(log_content)
                            logger.info("=" * 80)
                        else:
                            logger.info(f"Worker log file is empty: {log_path}")
                except Exception as e:
                    logger.warning(f"Failed to read worker log: {e}")
            else:
                logger.warning(f"Worker log file not found: {log_path}")
        except Exception as e:
            logger.warning(f"Failed to read log file: {e}")

    def run_cli_command(self, args, expect_success=True, timeout=3600):
        """
        Run mindformers_cli command and return result.
        This method waits for the subprocess to complete before returning.

        Args:
            args: List of command line arguments
            expect_success: Whether the command is expected to succeed (default: True)
            timeout: Command timeout in seconds (default: 3600 for training/inference)

        Returns:
            subprocess.CompletedProcess: The result of the command execution
        """
        cmd = ['mindformers-cli'] + args
        logger.info(f"Running command: {' '.join(cmd)}")

        env = os.environ.copy()
        # Clear any existing suffix (LOG_MF_PATH is deprecated)
        env.pop('MF_LOG_SUFFIX', None)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=False
            )

            logger.info(f"Command return code: {result.returncode}")
            if result.stdout:
                logger.info(f"Command stdout:\n{result.stdout}")
            if result.stderr:
                logger.info(f"Command stderr:\n{result.stderr}")

            if expect_success:
                assert result.returncode == 0, (
                    f"Command failed with return code {result.returncode}.\n"
                    f"Stdout: {result.stdout}\nStderr: {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            pytest.fail(f"Command timed out after {timeout} seconds")
        except FileNotFoundError:
            pytest.fail("'mindformers-cli' command not found. Please ensure mindformers is installed and "
                        "mindformers-cli is available in PATH")
        return result

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_pretrain_single_card(self):
        """
        Feature: Pretrain task in single card mode
        Description: Test pretrain task execution in single card mode using real config template.
                     This test waits for the training subprocess to complete.
        Expectation: Should successfully complete training in single card mode
        """
        assert self.pretrain_config_template_path.exists(), \
            f"Pretrain config template not found: {self.pretrain_config_template_path}"

        # Set output_dir to temporary directory
        output_dir_single = str(self.tmp_path / "output_single")

        # Build CLI args with all config overrides
        # Use json.dumps for list values to ensure proper JSON formatting
        data_path_json = json.dumps(['1.0', self.dataset_path])
        sizes_json = json.dumps([100, 0, 0])

        args = [
            '--config', str(self.pretrain_config_template_path),
            '--pretrained_model_dir', self.model_dir,
            '--trust_remote_code', 'True',
            '--use_parallel', 'False',
            '--output_dir', output_dir_single,
            '--train_dataset.data_loader.sizes', sizes_json,
            '--train_dataset.data_loader.config.data_path', data_path_json,
            '--train_dataset.data_loader.config.seq_length', '1024',
            '--training_args.global_batch_size', '16',
            '--checkpoint_config.load_checkpoint', 'not_load_any_ckpt',
            '--context.ascend_config.parallel_speed_up_json_path', self.parallel_speed_up_json_path
        ]
        # expect_success=True means the test will wait for subprocess to complete and verify success
        result = self.run_cli_command(args, expect_success=True)

        # Verify training completed successfully
        assert result.returncode == 0, (
            f"Training failed with return code {result.returncode}.\n"
            f"Stdout: {result.stdout}\nStderr: {result.stderr}"
        )

        # Check for unexpected errors
        self.check_unexpected_errors(result)

        # Verify command produced output
        output = result.stdout + result.stderr
        assert len(output) > 0, "Command produced no output"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_pretrain_multi_card_single_node(self):
        """
        Feature: Pretrain task in multi-card mode (single node)
        Description: Test pretrain task execution in multi-card mode on single node.
                     This test waits for the training subprocess to complete.
        Expectation: Should successfully complete training in multi-card mode on single node
        """
        assert self.pretrain_config_template_path.exists(), \
            f"Pretrain config template not found: {self.pretrain_config_template_path}"

        # Set log directory and output directory in temporary directory
        log_dir = str(self.tmp_path / "msrun_log")
        output_dir_multi = str(self.tmp_path / "output_multi")

        # Use json.dumps for list values to ensure proper JSON formatting
        data_path_json = json.dumps(['1.0', self.dataset_path])
        sizes_json = json.dumps([100, 0, 0])

        args = [
            '--worker-num', '2',  # Use 2 cards
            '--config', str(self.pretrain_config_template_path),
            '--master-port', '8119',
            '--log-dir', log_dir,
            '--join', 'True',  # Join existing cluster to wait for all workers to complete
            # Config overrides
            '--pretrained_model_dir', self.model_dir,
            '--trust_remote_code', 'True',
            '--use_parallel', 'True',
            '--output_dir', output_dir_multi,
            '--train_dataset.data_loader.sizes', sizes_json,
            '--train_dataset.data_loader.config.data_path', data_path_json,
            '--train_dataset.data_loader.config.seq_length', '1024',
            '--training_args.global_batch_size', '16',
            '--checkpoint_config.load_checkpoint', 'not_load_any_ckpt',
            '--context.ascend_config.parallel_speed_up_json_path', self.parallel_speed_up_json_path
        ]
        # expect_success=True means the test will wait for subprocess to complete and verify success
        result = self.run_cli_command(args, expect_success=True)

        # Verify training completed successfully
        assert result.returncode == 0, (
            f"Training failed with return code {result.returncode}.\n"
            f"Stdout: {result.stdout}\nStderr: {result.stderr}"
        )

        # Check for unexpected errors
        self.check_unexpected_errors(result)

        # Print worker_0.log from log_dir set via --log-dir
        self.print_worker_log(log_dir)

        # Verify command produced output
        output = result.stdout + result.stderr
        assert len(output) > 0, "Command produced no output"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_predict_single_card(self):
        """
        Feature: Predict task in single card mode
        Description: Test predict task execution in single card mode using real config template.
                     This test waits for the inference subprocess to complete.
        Expectation: Should successfully complete inference in single card mode
        """
        assert self.predict_config_template_path.exists(), \
            f"Predict config template not found: {self.predict_config_template_path}"

        # Set output_dir to temporary directory
        predict_output_dir_single = str(self.tmp_path / "output_predict_single")

        args = [
            '--config', str(self.predict_config_template_path),
            # Config overrides
            '--pretrained_model_dir', self.model_dir,
            '--trust_remote_code', 'True',
            '--use_parallel', 'False',
            '--output_dir', predict_output_dir_single,
            '--input_data', '<|im_start|>'
        ]
        # expect_success=True means the test will wait for subprocess to complete and verify success
        result = self.run_cli_command(args, expect_success=True)

        # Verify inference completed successfully
        assert result.returncode == 0, (
            f"Inference failed with return code {result.returncode}.\n"
            f"Stdout: {result.stdout}\nStderr: {result.stderr}"
        )

        # Check for unexpected errors
        self.check_unexpected_errors(result)

        # Verify command produced output
        output = result.stdout + result.stderr
        assert len(output) > 0, "Command produced no output"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_predict_multi_card_single_node(self):
        """
        Feature: Predict task in multi-card mode (single node)
        Description: Test predict task execution in multi-card mode on single node.
                     This test waits for the inference subprocess to complete.
        Expectation: Should successfully complete inference in multi-card mode on single node
        """
        assert self.predict_config_template_path.exists(), \
            f"Predict config template not found: {self.predict_config_template_path}"

        # Set log directory and output directory in temporary directory
        log_dir = str(self.tmp_path / "msrun_log")
        predict_output_dir_multi = str(self.tmp_path / "output_predict_multi")

        args = [
            '--worker-num', '2',  # Use 2 cards
            '--config', str(self.predict_config_template_path),
            '--master-port', '8120',
            '--log-dir', log_dir,
            '--join', 'True',  # Join existing cluster to wait for all workers to complete
            # Config overrides
            '--pretrained_model_dir', self.model_dir,
            '--trust_remote_code', 'True',
            '--use_parallel', 'True',
            '--distribute_parallel_config.tensor_model_parallel_size', '2',
            '--output_dir', predict_output_dir_multi,
            '--input_data', '<|im_start|>'
        ]
        # expect_success=True means the test will wait for subprocess to complete and verify success
        result = self.run_cli_command(args, expect_success=True)

        # Verify inference completed successfully
        assert result.returncode == 0, (
            f"Inference failed with return code {result.returncode}.\n"
            f"Stdout: {result.stdout}\nStderr: {result.stderr}"
        )

        # Check for unexpected errors
        self.check_unexpected_errors(result)

        # Print worker_0.log from log_dir set via --log-dir
        self.print_worker_log(log_dir)

        # Verify command produced output
        output = result.stdout + result.stderr
        assert len(output) > 0, "Command produced no output"
