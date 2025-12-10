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
"""Mindformers CLI"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

from mindformers.tools.register import MindFormerConfig
from mindformers.tools.utils import parse_value
from mindformers.core.context import build_context
from mindformers.utils.file_utils import set_output_path
from mindformers.trainer import build_trainer


def setup_environment():
    """Set up environment variables for logging and workspace.

    Returns:
        tuple: A tuple containing:
            - mf_log_suffix (str): Log suffix from environment variable
            - workspace_path (Path): Current working directory path
    """
    # Set workspace path
    workspace_path = Path.cwd()

    # Set log suffix
    mf_log_suffix = os.environ.get('MF_LOG_SUFFIX', '')
    if mf_log_suffix:
        mf_log_suffix = f"_{mf_log_suffix}"

    # Set PLOG path
    plog_redirect = os.environ.get('PLOG_REDIRECT_TO_OUTPUT', 'False')
    if plog_redirect.lower() == 'true':
        plog_path = workspace_path / f"output/plog{mf_log_suffix}"
        os.environ['ASCEND_PROCESS_LOG_PATH'] = str(plog_path)
        print(f"PLOG_REDIRECT_TO_OUTPUT={plog_redirect}, set the path of plog to {plog_path}")

    return mf_log_suffix, workspace_path


def apply_config_overrides(config, overrides):
    """Apply configuration overrides to config.
    
    Args:
        config: MindFormerConfig object
        overrides: List of override strings in format "key.path=value" or ["key.path", "value"]
    """
    for override in overrides:
        if '=' in override:
            # Format: key.path=value
            key, value = override.split('=', 1)
        else:
            raise ValueError(f"Invalid override format: {override}. Expected format: key.path=value")

        if not key.startswith("--"):
            raise ValueError(f"Config key must start with '--'. Got: {key}")

        # Remove '--' prefix and split by '.'
        config_path = key[2:].split(".")

        # Use set_value to set nested config values
        config.set_value(config_path, parse_value(value))


def main():
    """Single card task entry point.

    Executes single card training or inference tasks based on the provided configuration.

    Raises:
        ValueError: If run_mode is not supported.
    """
    parser = argparse.ArgumentParser(description='Single card training/inference task')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')

    args, unknown_args = parser.parse_known_args()

    work_path = os.path.dirname(os.path.abspath(__file__))
    if args.config is not None and not os.path.isabs(args.config):
        args.config = os.path.join(work_path, args.config)

    # Setting Environment Variables: USE_CONFIG_TEMPLATE_V2 For New Config Template
    os.environ["USE_CONFIG_TEMPLATE_V2"] = "1"

    config = MindFormerConfig(args.config)

    # Apply configuration overrides from unknown arguments
    if unknown_args:
        # Process unknown args: split by '=' or treat as key-value pairs
        overrides = []
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]
            if '=' in arg:
                # Format: --key.path=value
                overrides.append(arg)
            elif arg.startswith('--') and i + 1 < len(unknown_args):
                # Format: --key.path value
                overrides.append(f"{arg}={unknown_args[i+1]}")
                i += 1  # Skip next arg as it's the value
            else:
                raise ValueError(f"Invalid argument format: {arg}. "
                                 f"Expected format: --key.path=value or --key.path value")
            i += 1

        apply_config_overrides(config, overrides)

    if config.register_path is not None:
        if not os.path.isabs(config.register_path):
            config.register_path = os.path.join(work_path, args.register_path)
        # Setting Environment Variables: REGISTER_PATH For Auto Register to Outer API
        os.environ["REGISTER_PATH"] = config.register_path
        if config.register_path not in sys.path:
            sys.path.append(config.register_path)

    # set output path
    set_output_path(config.output_dir)

    # init context
    build_context(config)

    trainer = build_trainer(config.trainer)
    if config.run_mode in ['train', 'finetune']:
        trainer.train(config)
    elif config.run_mode == 'predict':
        trainer.predict(config)
    else:
        raise ValueError(
            f"Run mode {config.run_mode} is not supported. Please select from 'train', 'predict' or 'finetune'.")


def create_argument_parser():
    """Create and configure argument parser for launcher.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Distributed task launcher using msrun',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single card training
  mindformers-cli --config config.yaml

  # Single node multi-card (8 cards)
  mindformers-cli --worker-num 8 --config config.yaml

  # Single node multi-card with custom parameters
  mindformers-cli --worker-num 4 --master-port 8080 --log-dir output/my_log --config config.yaml

  # Multi-node multi-card
  mindformers-cli --worker-num 16 --local-worker 8 --master-addr 192.168.1.100 --node-rank 0 --config config.yaml

  # Override config values (supports both formats)
  mindformers-cli --config config.yaml --distribute_parallel_config.tensor_model_parallel_size=2
  mindformers-cli --config config.yaml --distribute_parallel_config.tensor_model_parallel_size 2
        """
    )

    # Distributed configuration parameters
    dist_group = parser.add_argument_group('Distributed Training Options')
    dist_group.add_argument('--worker-num', type=int, default=1,
                            help='Total number of workers (default: 1)')
    dist_group.add_argument('--local-worker', type=int,
                            help='Number of local workers (default: same as worker-num)')
    dist_group.add_argument('--master-addr', type=str, default='127.0.0.1',
                            help='Master node address (default: 127.0.0.1)')
    dist_group.add_argument('--master-port', type=int, default=8118,
                            help='Master node port (default: 8118)')
    dist_group.add_argument('--node-rank', type=int, default=0,
                            help='Node rank in distributed training (default: 0)')
    dist_group.add_argument('--log-dir', type=str, default='output/msrun_log',
                            help='Log directory (default: output/msrun_log)')
    dist_group.add_argument('--join', type=str, default='False',
                            help='Whether to join existing cluster (default: False)')
    dist_group.add_argument('--cluster-time-out', type=int, default=7200,
                            help='Cluster timeout in seconds (default: 7200)')
    dist_group.add_argument('--bind-core', type=str, default="True",
                            help='CPU binding configuration. Set to "True" for automatic allocation based on device'
                                 ' affinity, or provide a dictionary string like '
                                 '\'{"device0":["0-10"],"device1":["11-20"]}\' '
                                 'for manual CPU range assignment. If not set, defaults to True (automatic allocation)')

    # Task configuration parameters
    task_group = parser.add_argument_group('Task Options')
    task_group.add_argument('--config', type=str, required=True,
                            help='Configuration file path')
    # Note: Custom config overrides can be passed as unknown arguments
    # Format: --key.path=value or --key.path value
    # Example: --distribute_parallel_config.tensor_model_parallel_size=2

    return parser


def handle_single_card_mode(args, unknown_args):
    """Handle single card mode execution.
    
    Args:
        args: Parsed arguments
        unknown_args: Unknown arguments (config overrides)
    
    Returns:
        bool: True if single card mode was handled, False otherwise
    """
    if args.worker_num == 1:
        print("Running in single card mode...")
        # Directly call main function
        # Pass unknown args (config overrides) to main function
        sys.argv = [sys.argv[0]] + ['--config', args.config] + unknown_args
        main()
        return True
    return False


def build_task_args(args, unknown_args):
    """Build task arguments to pass to subprocess.

    Args:
        args: Parsed arguments
        unknown_args: Unknown arguments (config overrides)
    
    Returns:
        list: List of task arguments
    """
    task_args = ['--config', args.config]
    # Add unknown args (config overrides) to task args
    task_args.extend(unknown_args)
    return task_args


def build_msrun_command(args, log_dir, task_args):
    """Build msrun command with all required parameters.

    Args:
        args: Parsed arguments
        log_dir: Log directory with suffix
        task_args: Task arguments to pass to subprocess
    
    Returns:
        list: Complete msrun command as list
    """
    msrun_cmd = [
        'msrun',
        f'--worker_num={args.worker_num}',
        f'--local_worker_num={args.local_worker}',
        f'--master_port={args.master_port}',
        f'--log_dir={log_dir}',
        f'--join={args.join}',
        f'--cluster_time_out={args.cluster_time_out}'
    ]

    # Handle bind_core parameter
    if args.bind_core is not None:
        if args.bind_core.lower() == 'true':
            msrun_cmd.append('--bind_core=True')
        elif args.bind_core.lower() == 'false':
            msrun_cmd.append('--bind_core=False')
        else:
            # Assume it's a dictionary string, pass it as is
            msrun_cmd.append(f'--bind_core={args.bind_core}')

    # If multi-node training, add additional parameters
    if args.worker_num != args.local_worker:
        msrun_cmd.extend([
            f'--master_addr={args.master_addr}',
            f'--node_rank={args.node_rank}'
        ])
        print(f"Multi-node training: worker_num={args.worker_num}, local_worker={args.local_worker}")
    else:
        print(f"Single-node training: worker_num={args.worker_num}")

    # Add command to execute (main function of current script)
    msrun_cmd.extend([__file__] + task_args)

    return msrun_cmd


def setup_resource_limits():
    """Set up resource limits (ulimit) for Unix systems."""
    try:
        # pylint: disable=import-outside-toplevel
        import resource
        resource.setrlimit(resource.RLIMIT_NPROC, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ImportError, ValueError):
        print("Note: ulimit adjustment not available on this system")


def execute_msrun_command(msrun_cmd, workspace_path, log_dir):
    """Execute msrun command and handle errors.

    Args:
        msrun_cmd: Complete msrun command as list
        workspace_path: Workspace path for log file location
        log_dir: Log directory path

    Raises:
        SystemExit: If msrun command fails or is not found.
    """
    print(f"Running Command: {' '.join(msrun_cmd)}")
    print(f"Please check log files in {workspace_path / log_dir}")

    try:
        subprocess.run(msrun_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'msrun' command not found. Please make sure it's installed and in your PATH.")
        sys.exit(1)


def launcher():
    """Task launcher - command line tool entry point.

    This function is registered as a command via entry_point and handles
    standalone or distributed task.

    Raises:
        SystemExit: If msrun command fails or is not found.
    """
    mf_log_suffix, workspace_path = setup_environment()

    parser = create_argument_parser()
    args, unknown_args = parser.parse_known_args()

    # Set default local_worker
    if args.local_worker is None:
        args.local_worker = args.worker_num

    # Add log suffix
    log_dir = f"{args.log_dir}{mf_log_suffix}"

    # Check if running in single card mode
    if handle_single_card_mode(args, unknown_args):
        return

    # Build task arguments
    task_args = build_task_args(args, unknown_args)

    # Build msrun command
    msrun_cmd = build_msrun_command(args, log_dir, task_args)

    # Set ulimit (Unix systems)
    setup_resource_limits()

    # Execute command
    execute_msrun_command(msrun_cmd, workspace_path, log_dir)


if __name__ == "__main__":
    # Directly execute single card task
    main()
