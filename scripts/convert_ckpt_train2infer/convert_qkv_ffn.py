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
""" Parallel convert qkv and ffn matmul """

import os
from datetime import datetime
import multiprocessing
import argparse
import mindspore as ms
from mindformers.models.llama.convert_weight import convert_qkv_concat_weight
# from convert_weight import transpose_w2_weight


def add_qkv(i, src_ckpt_path, dst_ckpt_path):
    """convert previous ckpt to qkv concat ckpt"""
    rank_id = int(i)
    src_path = src_ckpt_path + "/rank_{}/".format(rank_id)
    dst_path = dst_ckpt_path + "/rank_{}/".format(rank_id)
    ckpt_name = os.listdir(src_path)[0]
    params = ms.load_checkpoint(src_path + ckpt_name)
    params = convert_qkv_concat_weight(params)
    # if w2_transb:
    #     params = transpose_w2_weight(params)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    ms.save_checkpoint(params, dst_path + ckpt_name)


def main(src_ckpt_path, dst_ckpt_path, world_size):
    """parallel run add_qkv function"""
    # 获取当前时间
    start_time = datetime.now().strftime("%H:%M:%S")

    arguments = [(i, src_ckpt_path, dst_ckpt_path) for i in range(world_size)]

    # 创建一个进程池
    with multiprocessing.Pool(processes=world_size) as pool:
        # 使用pool.starmap并行执行transform_ckpt函数，传入参数列表
        pool.starmap(add_qkv, arguments)

    # 获取结束时间
    end_time = datetime.now().strftime("%H:%M:%S")

    # 打印开始和结束时间
    print(f"add qkv ckpt: start time: {start_time}, End time: {end_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt_path', default='', type=str,
                        help='Checkpoint saved path from train process.')
    parser.add_argument('--dst_ckpt_path', default='', type=str,
                        help='Checkpoint saved path from train process.')
    parser.add_argument('--world_size', default=8, type=int,
                        help='world size')
    args = parser.parse_args()

    main(args.src_ckpt_path, args.dst_ckpt_path, args.world_size)
