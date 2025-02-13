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
"""
merge safetensors weight.
Support mindspore format.
"""
import time
import argparse
import os.path
import shutil
import mindspore as ms
from mindformers.tools.utils import str2bool
from mindformers.tools.logger import logger


def unified_safetensors(src_dir, src_merge_strategy, output_dir, file_suffix, has_redundancy):
    """merge safetensors files."""
    _timed_print("Start merge safetensor")
    merged_path_ = os.path.join(output_dir, "unified_safe")
    if not os.path.exists(merged_path_):
        os.makedirs(merged_path_, exist_ok=True)
    ms.unified_safetensors(src_dir, src_merge_strategy, merged_path_,
                           file_suffix=file_suffix, merge_with_redundancy=has_redundancy)
    _timed_print("Merge safetensor completed")
    return merged_path_


def merge_pipeline_strategys(src_strategy_dirs, output_dir):
    """merge pipeline strategys."""
    _timed_print("Start merge strategy")
    dst_strategy_dir = os.path.join(output_dir, "merge_strategy")
    os.makedirs(dst_strategy_dir, exist_ok=True)
    dst_strategy_file = os.path.join(dst_strategy_dir, "merge_strategy.ckpt")
    ms.merge_pipeline_strategys(src_strategy_dirs, dst_strategy_file)
    _timed_print("Merge strategy completed")
    return dst_strategy_file


def clear_output_dir(output_dir):
    """clear files."""
    if os.path.exists(output_dir):
        _timed_print(f"Start clear tmp file: {output_dir}")
        shutil.rmtree(output_dir)
        _timed_print("Clear tmp file completed")
    return True


def _timed_print(args):
    """timed print."""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + args
    logger.info(f"[{current_time}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mindspore_ckpt_dir', default='/path/checkpoint')
    parser.add_argument('--src_strategy_dirs', default='/path/strategy')
    parser.add_argument('--output_dir', default='/path/output_dir')
    parser.add_argument('--file_suffix', default='1_1', help="1_1 is suffix of checkpoint. If ckpt file:"
                                                             "llama3_31b_rank_0-1_1.ckpt, file_suffix is: 1_1")
    parser.add_argument('--format', default="safetensors", choices=["ckpt", "safetensors"])
    parser.add_argument('--has_redundancy', default=True, type=str2bool,
                        choices=[True, False], help='whether input ckpt file has redundancy')
    _args = parser.parse_args()
    _args.output_dir = os.path.join(_args.output_dir, str(_args.file_suffix) + "_ckpt_convert")

    # Print arguments
    _timed_print("args config:")
    for k, v in sorted(vars(_args).items()):
        logger.info(f"{k}={v}")
    start_time = time.time()
    _timed_print("Task start...")

    clear_output_dir(_args.output_dir)
    merge_strategy_file = merge_pipeline_strategys(_args.src_strategy_dirs, _args.output_dir)
    merged_path = unified_safetensors(_args.mindspore_ckpt_dir, merge_strategy_file,
                                      _args.output_dir, _args.file_suffix, _args.has_redundancy)

    logger.info(f'merged safetensor path: {merged_path}')

    # time show
    end_time = time.time()
    _timed_print("Task completed time:")
    _timed_print(f"Task total cost time: {end_time - start_time}s")
