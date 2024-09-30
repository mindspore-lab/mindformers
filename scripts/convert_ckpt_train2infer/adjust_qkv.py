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

import os
import argparse
from datetime import datetime
import multiprocessing
import numpy as np
import mindspore as ms


def adjust_single_param(params_dict, param_name, group, is_qkv):
    """Adjust single param from checkpoint"""
    if param_name not in params_dict:
        return False
    print(f"Processing {param_name}...", flush=True)
    param = params_dict[param_name].asnumpy()
    total = param.shape[0]
    group_members = 3 if is_qkv else 2
    segment = total // group // group_members
    member0 = []
    member1 = []
    member2 = []
    for j in range(group):
        p0 = (j * group_members + 0) * segment
        p1 = (j * group_members + 1) * segment
        p2 = (j * group_members + 2) * segment
        member0.append(param[p0:p1,])
        member1.append(param[p1:p2,])
        if is_qkv:
            p3 = (j * group_members + 3) * segment
            member2.append(param[p2:p3,])
    if is_qkv:
        orderd_list = member0 + member1 + member2
    else:  # ffn
        orderd_list = member0 + member1
    params_dict[param_name] = ms.Parameter(np.concatenate(orderd_list, 0), name=param_name)
    return True


def adjust_single_ckpt(src_ckpt_file, dst_ckpt_file, src_tp=4, dst_tp=2, quant='True'):
    """Adjust qkv"""
    group = src_tp // dst_tp
    if group == 0:
        raise ValueError(f"Invalid src_tp({src_tp}) and dst_tp({dst_tp}).")
    print(f"Loading {src_ckpt_file}...", flush=True)
    params_dict = ms.load_checkpoint(src_ckpt_file)
    changed = False
    i = 0
    if quant == 'True':
        while True:
            changed = False
            # qkv weight adjust for a8a16
            qkv_weight_name = f"model.layers.{i}.attention.w_qkv._layer.weight"
            changed |= adjust_single_param(params_dict, qkv_weight_name, group, True)
            # qkv bias adjust for a8
            qkv_bias_name = f"model.layers.{i}.attention.w_qkv._layer.bias"
            changed |= adjust_single_param(params_dict, qkv_bias_name, group, True)
            # qkv matmul dequant scale adjust for a8
            qkv_mds_name = f"model.layers.{i}.attention.w_qkv._layer.matmul.dequant_scale"
            changed |= adjust_single_param(params_dict, qkv_mds_name, group, True)
            # qkv matmul t scale adjust for a16
            qkv_mts_name = f"model.layers.{i}.attention.w_qkv._layer.matmul.t_scale"
            changed |= adjust_single_param(params_dict, qkv_mts_name, group, True)
            # ffn weight adjust for a8a16
            ffn_weight_name = f"model.layers.{i}.feed_forward.w_gate_hidden._layer.weight"
            changed |= adjust_single_param(params_dict, ffn_weight_name, group, False)
            # ffn bias adjust for a8
            ffn_bias_name = f"model.layers.{i}.feed_forward.w_gate_hidden._layer.bias"
            changed |= adjust_single_param(params_dict, ffn_bias_name, group, False)
            # ffn matmul dequant scale adjust for a8
            ffn_mds_name = f"model.layers.{i}.feed_forward.w_gate_hidden._layer.matmul.dequant_scale"
            changed |= adjust_single_param(params_dict, ffn_mds_name, group, False)
            # ffn matmul t scale adjust for a16
            ffn_mts_name = f"model.layers.{i}.feed_forward.w_gate_hidden._layer.matmul.t_scale"
            changed |= adjust_single_param(params_dict, ffn_mts_name, group, False)
            if changed:
                i += 1
            else:
                break
    else:
        while True:
            changed = False
            # qkv weight adjust
            qkv_weight_name = f"model.layers.{i}.attention.w_qkv.weight"
            changed |= adjust_single_param(params_dict, qkv_weight_name, group, True)
            # ffn weight adjust
            ffn_weight_name = f"model.layers.{i}.feed_forward.w_gate_hidden.weight"
            changed |= adjust_single_param(params_dict, ffn_weight_name, group, False)
            if changed:
                i += 1
            else:
                break
    ms.save_checkpoint(params_dict, dst_ckpt_file)
    print(f"Saved ckpt file: {dst_ckpt_file}.", flush=True)


def run_adjust_qkv(i, src_ckpt_path, dst_ckpt_path, src_tp, dst_tp, quant):
    """Run adjust qkv"""
    rank_id = int(i)
    src_path = os.path.join(src_ckpt_path, f"rank_{rank_id}")
    dst_path = os.path.join(dst_ckpt_path, f"rank_{rank_id}")
    ckpt_name = os.listdir(src_path)[0]
    src_ckpt_file = os.path.join(src_path, ckpt_name)
    os.makedirs(dst_path, exist_ok=True)
    dst_ckpt_file = os.path.join(dst_path, ckpt_name)
    adjust_single_ckpt(src_ckpt_file, dst_ckpt_file, src_tp, dst_tp, quant)


def main(args):
    """Parallel run run_adjust_qkv"""
    # 获取当前时间
    start_time = datetime.now().strftime("%H:%M:%S")
    if (args.dir_count == 8 and args.world_size == 4) or (args.dir_count == 4 and args.world_size == 2):
        arguments = [(i, args.src_ckpt_path, args.dst_ckpt_path, args.dir_count, args.world_size, args.quant) for i in
                     range(args.world_size)]
        # 创建一个进程池
        with multiprocessing.Pool(processes=args.world_size) as pool:
            # 使用pool.starmap并行执行transform_ckpt函数，传入参数列表
            pool.starmap(run_adjust_qkv, arguments)
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
    parser.add_argument('--quant', default='True', type=str,
                        help='Weight is quant or not')
    uargs = parser.parse_args()

    main(uargs)
