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
import numpy as np
import mindspore as ms


def convert_qkv_concat_weight(param_dict):
    """convert qkv concat weight"""
    assume_num_layers = 500
    for i in range(assume_num_layers):
        # qkv weight concat
        wq_weight_name = f"model.layers.{i}.attention.wq.weight"
        wk_weight_name = f"model.layers.{i}.attention.wk.weight"
        wv_weight_name = f"model.layers.{i}.attention.wv.weight"
        qkv_concat_weight_name = f"model.layers.{i}.attention.w_qkv.weight"
        if wq_weight_name not in param_dict:
            break
        wq_weight = param_dict[wq_weight_name].asnumpy()
        wk_weight = param_dict[wk_weight_name].asnumpy()
        wv_weight = param_dict[wv_weight_name].asnumpy()
        qkv_weight = np.concatenate((wq_weight, wk_weight, wv_weight), 0)
        param_dict[qkv_concat_weight_name] = ms.Parameter(qkv_weight, name=qkv_concat_weight_name)

        # gate hidden weight concat
        ffn_gate_weight_name = f"model.layers.{i}.feed_forward.w1.weight"
        ffn_hidden_weight_name = f"model.layers.{i}.feed_forward.w3.weight"
        gate_hidden_concat_weight_name = f"model.layers.{i}.feed_forward.w_gate_hidden.weight"

        ffn_gate_weight = param_dict[ffn_gate_weight_name].asnumpy()
        ffn_hidden_weight = param_dict[ffn_hidden_weight_name].asnumpy()
        gate_hidden_weight = np.concatenate((ffn_gate_weight, ffn_hidden_weight), 0)
        param_dict[gate_hidden_concat_weight_name] = ms.Parameter(gate_hidden_weight,
                                                                  name=gate_hidden_concat_weight_name)

        param_dict.pop(wq_weight_name)
        param_dict.pop(wk_weight_name)
        param_dict.pop(wv_weight_name)
        param_dict.pop(ffn_gate_weight_name)
        param_dict.pop(ffn_hidden_weight_name)
        print("transform: {}".format(qkv_concat_weight_name))
        print("transform: {}".format(gate_hidden_concat_weight_name))

    for i in range(assume_num_layers):
        # qkv bias concat
        wq_bias_name = f"model.layers.{i}.attention.wq.bias"
        wk_bias_name = f"model.layers.{i}.attention.wk.bias"
        wv_bias_name = f"model.layers.{i}.attention.wv.bias"
        qkv_concat_bias_name = f"model.layers.{i}.attention.w_qkv.bias"
        if wq_bias_name not in param_dict:
            break

        wq_bias_weight = param_dict[wq_bias_name].asnumpy()
        wk_bias_weight = param_dict[wk_bias_name].asnumpy()
        wv_bias_weight = param_dict[wv_bias_name].asnumpy()
        qkv_bias_weight = np.concatenate((wq_bias_weight, wk_bias_weight, wv_bias_weight), 0)
        param_dict[qkv_concat_bias_name] = ms.Parameter(qkv_bias_weight, name=qkv_concat_bias_name)

        param_dict.pop(wq_bias_name)
        param_dict.pop(wk_bias_name)
        param_dict.pop(wv_bias_name)
        print("transform: {}".format(qkv_concat_bias_name))
    return param_dict


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
