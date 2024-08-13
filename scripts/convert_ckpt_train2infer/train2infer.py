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
""""Convert train ckpt to infer ckpt"""

import time
import copy
import os
from datetime import datetime
import multiprocessing
import argparse
import mindspore as ms


def del_optimize_param(i, train_ckpt_path, del_optim_path):
    """Delete optimize parameter in training checkpoint"""
    t1 = time.time()
    ms.set_context(device_target="CPU")
    rank_id = int(i)
    src_path = train_ckpt_path + "/rank_{}/".format(rank_id)
    dst_path = del_optim_path + "/rank_{}/".format(rank_id)

    ckpt_name = os.listdir(src_path)[0]

    params = ms.load_checkpoint(src_path + ckpt_name)
    os.makedirs(dst_path)

    keys = copy.deepcopy(list(params.keys()))
    save_keys = []
    for name in keys:
        if not name.startswith("model") and name != "lm_head.weight":
            del params[name]
        else:
            params[name] = ms.Parameter(params[name].astype(ms.float16))
            save_keys.append(name)

    ms.save_checkpoint(params, dst_path + ckpt_name)
    t2 = time.time()
    print(f"del total time: {t2 - t1}")


def find_divisors(n):
    divisors = set()
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return sorted(divisors)


def run_del(train_ckpt_path, del_optim_path, train_folder_count):
    """parallel run del_optimize_param function"""
    # 获取当前时间
    start_time = datetime.now().strftime("%H:%M:%S")

    cpu_count = multiprocessing.cpu_count()

    if cpu_count >= train_folder_count:
        arguments = [(i, train_ckpt_path, del_optim_path) for i in range(train_folder_count)]
        # 创建一个进程池，指定进程数量
        with multiprocessing.Pool(processes=train_folder_count) as pool:
            pool.starmap(del_optimize_param, arguments)
    else:
        divisors = find_divisors(train_folder_count)
        # 找到最接近但不超过CPU核心数*2的最大整除数
        process_num = max(d for d in divisors if d <= cpu_count * 2)
        # 定义要处理的数据列表
        num_chunks = train_folder_count // process_num  # 3584 // 256
        # 使用pool.map并行执行worker函数，传入每个输入值
        for j in range(num_chunks):
            start_chunk_time = datetime.now().strftime("%H:%M:%S")
            print(f"Start chunk {j + 1}/{num_chunks} at {start_chunk_time}")
            arguments = [(i, train_ckpt_path, del_optim_path) for i in
                         range(j * process_num, min((j + 1) * process_num, train_folder_count))]
            # 创建一个进程池，指定进程数量
            with multiprocessing.Pool(processes=process_num) as pool:
                pool.starmap(del_optimize_param, arguments)

            end_chunk_time = datetime.now().strftime("%H:%M:%S")
            print(f"End chunk {j + 1}/{num_chunks} at {end_chunk_time}")

    # 获取结束时间
    end_time = datetime.now().strftime("%H:%M:%S")

    # 打印开始和结束时间
    print(f"delete optimize: start time: {start_time}, End time: {end_time}")


def transform_ckpt(i, del_optim_path, save_checkpointint_file_dir, train_strategy_file, infer_strategy_file,
                   pipeline_stage, train_folder_count):
    """transform training checkpoint to infer checkpoint"""
    ms.set_context(device_target="CPU")
    stage_id = int(i)
    stage_device_num = train_folder_count // pipeline_stage  # 512 3584/7   #256 3840/15

    train_strategy_file = train_strategy_file + "/ckpt_strategy_rank_{}.ckpt".format(stage_id * stage_device_num)
    infer_strategy_file = infer_strategy_file + "/ckpt_strategy_rank_0.ckpt"
    src_checkpoint_file_dir = del_optim_path
    save_checkpointint_file_dir = save_checkpointint_file_dir

    start_time = time.time()
    ms.transform_checkpoints(src_checkpoint_file_dir, save_checkpointint_file_dir, "checkpoint_", train_strategy_file,
                             infer_strategy_file)
    print("transform rank {} cost: {:.2f}".format(stage_id, time.time() - start_time))


def run_transform(del_optim_path, train_2_infer_path, train_strategy_file, infer_strategy_file, pipeline_stage,
                  train_folder_count):
    """parallel run transform_ckpt function"""
    # 获取当前时间
    start_time = datetime.now().strftime("%H:%M:%S")

    arguments = [(i, del_optim_path, train_2_infer_path, train_strategy_file, infer_strategy_file, pipeline_stage,
                  train_folder_count) for i in range(pipeline_stage)]

    # 创建一个进程池
    with multiprocessing.Pool(processes=pipeline_stage) as pool:
        # 使用pool.starmap并行执行transform_ckpt函数，传入参数列表
        pool.starmap(transform_ckpt, arguments)

    # 获取结束时间
    end_time = datetime.now().strftime("%H:%M:%S")

    # 打印开始和结束时间
    print(f"transform ckpt: start time: {start_time}, End time: {end_time}")


def merge(i, train_2_infer_path, infer_ckpt_path):
    """merge checkpoint"""
    rank_id = int(i)
    os.makedirs(infer_ckpt_path + "/rank_{}".format(rank_id))
    t1 = time.time()
    param_dict = ms.load_segmented_checkpoints(train_2_infer_path + "/rank_{}".format(rank_id))
    t2 = time.time()
    ms.save_checkpoint(param_dict, infer_ckpt_path + f"/rank_{rank_id}/llama_57B_rank_{rank_id}.ckpt")
    t3 = time.time()
    print(f"load time: {t2 - t1}, save time: {t3 - t2}, total time: {t3 - t1}")


def run_merge(train_2_infer_path, infer_ckpt_path, world_size):
    """parallel run merge function"""
    # 获取当前时间
    start_time = datetime.now().strftime("%H:%M:%S")
    arguments = [(i, train_2_infer_path, infer_ckpt_path) for i in range(world_size)]

    # 创建一个进程池
    with multiprocessing.Pool(processes=world_size) as pool:
        # 使用pool.starmap并行执行transform_ckpt函数，传入参数列表
        pool.starmap(merge, arguments)
    # 获取结束时间
    end_time = datetime.now().strftime("%H:%M:%S")

    # 打印开始和结束时间
    print(f"merge ckpt: start time: {start_time}, End time: {end_time}")


def main(args):
    """main function"""
    # 获取目录中的所有条目
    entries = os.listdir(args.train_ckpt_path)
    # 过滤出文件夹数量
    train_folder_count = sum(1 for entry in entries if
                             os.path.isdir(os.path.join(args.train_ckpt_path, entry)) and entry.startswith('rank_'))
    run_del(args.train_ckpt_path, args.del_optim_path, train_folder_count)
    run_transform(args.del_optim_path, args.train_2_infer_path, args.train_strategy_file, args.infer_strategy_file,
                  args.pipeline_stage, train_folder_count)
    run_merge(args.train_2_infer_path, args.infer_ckpt_path, args.world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ckpt_path', default='', type=str,
                        help='Checkpoint saved path from train process.')
    parser.add_argument('--del_optim_path', default='', type=str,
                        help='set device id.')
    parser.add_argument('--train_strategy_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--infer_strategy_file', default='', type=str,
                        help='Checkpoint saved path from train process.')
    parser.add_argument('--train_2_infer_path', default='', type=str,
                        help='Checkpoint saved path from train process.')
    parser.add_argument('--infer_ckpt_path', default='', type=str,
                        help='Checkpoint saved path from train process.')
    parser.add_argument('--world_size', default=8, type=int,
                        help='world size')
    parser.add_argument('--pipeline_stage', default=7, type=int,
                        help='training pipeline_stage')
    uargs = parser.parse_args()

    main(uargs)
