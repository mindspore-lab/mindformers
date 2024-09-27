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
""""Revert qkv and ffn"""
import os
from datetime import datetime
import multiprocessing
import argparse
import mindspore as ms


def revert_qkv_concat_weight(param_dict):
    """revert qkv concat weight"""
    assume_num_layers = 500
    for i in range(assume_num_layers):
        # qkv weight concat
        wq_weight_name = f"model.layers.{i}.attention.wq.weight"
        wk_weight_name = f"model.layers.{i}.attention.wk.weight"
        wv_weight_name = f"model.layers.{i}.attention.wv.weight"
        qkv_concat_weight_name = f"model.layers.{i}.attention.w_qkv.weight"
        if qkv_concat_weight_name not in param_dict:
            break
        value = param_dict[qkv_concat_weight_name]
        q, k, v = ms.ops.split(value, value.shape[0] // 3, axis=0)
        param_dict[wq_weight_name] = ms.Parameter(q, wq_weight_name)
        param_dict[wk_weight_name] = ms.Parameter(k, wk_weight_name)
        param_dict[wv_weight_name] = ms.Parameter(v, wv_weight_name)
        param_dict.pop(qkv_concat_weight_name)

        # gate hidden weight concat
        ffn_gate_weight_name = f"model.layers.{i}.feed_forward.w1.weight"
        ffn_hidden_weight_name = f"model.layers.{i}.feed_forward.w3.weight"
        gate_hidden_concat_weight_name = f"model.layers.{i}.feed_forward.w_gate_hidden.weight"
        value = param_dict[gate_hidden_concat_weight_name]
        gate, hidden = ms.ops.split(value, value.shape[0] // 2, axis=0)
        param_dict[ffn_gate_weight_name] = ms.Parameter(gate, ffn_gate_weight_name)
        param_dict[ffn_hidden_weight_name] = ms.Parameter(hidden, ffn_hidden_weight_name)
        param_dict.pop(gate_hidden_concat_weight_name)

    return param_dict


def remove_qkv(i, src_ckpt_path, dst_ckpt_path):
    """convert qkv concat ckpt to no-qkv concat ckpt"""
    rank_id = int(i)
    src_path = src_ckpt_path + "/rank_{}/".format(rank_id)
    dst_path = dst_ckpt_path + "/rank_{}/".format(rank_id)
    ckpt_name = os.listdir(src_path)[0]
    params = ms.load_checkpoint(src_path + ckpt_name)
    params = revert_qkv_concat_weight(params)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    ms.save_checkpoint(params, dst_path + ckpt_name)


def main(src_ckpt_path, dst_ckpt_path, world_size):
    """parallel run remove_qkv function"""
    # 获取当前时间
    ms.set_context(device_target='CPU')
    start_time = datetime.now().strftime("%H:%M:%S")

    arguments = [(i, src_ckpt_path, dst_ckpt_path) for i in range(world_size)]

    # 创建一个进程池
    with multiprocessing.Pool(processes=world_size) as pool:
        # 使用pool.starmap并行执行transform_ckpt函数，传入参数列表
        pool.starmap(remove_qkv, arguments)

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
