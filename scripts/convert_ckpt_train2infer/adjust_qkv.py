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
""" adjust qkv layout for quant transform different card numbers """

import argparse
from datetime import datetime
from mindformers.models.llama.convert_weight import adjust_quant_qkv_concat


def main(args):
    """Adjust qkv layout"""
    # 获取当前时间
    start_time = datetime.now().strftime("%H:%M:%S")
    if (args.dir_count == 8 and args.world_size == 4) or (args.dir_count == 4 and args.world_size == 2):
        adjust_quant_qkv_concat(args.src_ckpt_path, args.dst_ckpt_path, args.dir_count, args.world_size)
    else:
        raise ValueError(f"Invalid input: {args.dir_count} and {args.world_size}. Available: 4to2, 8to4.")
    # 获取结束时间
    end_time = datetime.now().strftime("%H:%M:%S")
    # 打印开始和结束时间
    print(f"add qkv ckpt: start time: {start_time}, End time: {end_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt_path', default='', type=str,
                        help='Checkpoint saved path.')
    parser.add_argument('--dst_ckpt_path', default='', type=str,
                        help='Path to save new checkpoint.')
    parser.add_argument('--world_size', type=int,
                        help='dst card number.')
    parser.add_argument('--dir_count', type=int,
                        help='src card number.')
    uargs = parser.parse_args()

    main(uargs)
