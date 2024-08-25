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
import mindspore as ms


def main(args):
    """transform ckpt to safetensors"""
    start_time = datetime.now().strftime("%H:%M:%S")
    ms.ckpt_to_safetensors(file_path=args.src_ckpt_path, save_path=args.dst_safetensors_path)
    end_time = datetime.now().strftime("%H:%M:%S")
    # 打印开始和结束时间
    print(f"add qkv ckpt: start time: {start_time}, End time: {end_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt_path', default='', type=str,
                        help='ckpt path.')
    parser.add_argument('--dst_safetensors_path', default='', type=str,
                        help='safetensors path.')
    uargs = parser.parse_args()

    main(uargs)
