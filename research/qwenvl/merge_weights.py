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

import argparse
import os

import numpy as np
from mindspore import load_checkpoint, save_checkpoint, Parameter


def merge_positional_embedding(shard_ckpt_dir, target_ckpt_path):
    """
    merge vision_encoder.positional_embedding from the distributed weights and merge it into the target ckpt
    """
    weight_name = "vision_encoder.positional_embedding"
    rank_dirs = os.listdir(shard_ckpt_dir)
    rank_dirs.sort()

    shard_ckpt_path_list = []
    for rank_dirname in rank_dirs:
        rank_dir = os.path.join(shard_ckpt_dir, rank_dirname)
        ckpt_names = [filename for filename in os.listdir(rank_dir) if filename.endswith(".ckpt")]
        ckpt_names.sort()
        shard_ckpt_path_list.append(os.path.join(rank_dir, ckpt_names[-1]))

    positional_embedding = []
    for shard_ckpt_path in shard_ckpt_path_list:
        ckpt = load_checkpoint(shard_ckpt_path)
        positional_embedding.append(ckpt.get(weight_name).asnumpy())

    del ckpt

    positional_embedding = Parameter(np.concatenate(positional_embedding, axis=0), name=weight_name)
    target_ckpt = load_checkpoint(target_ckpt_path)
    target_ckpt[weight_name] = positional_embedding

    ckpt_path = target_ckpt_path.replace(".ckpt", "_merge_pos_embedding.ckpt")
    save_checkpoint(target_ckpt, ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Qwen-VL merge weights")
    parser.add_argument("--shard_ckpt_dir", required=True, help="The dir of shard ckpts located.")
    parser.add_argument("--target_ckpt_path", required=True, help="The path of ckpt to be merged.")

    args = parser.parse_args()

    merge_positional_embedding(args.shard_ckpt_dir, args.target_ckpt_path)
