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
"""convert weight for moe infer ckpt"""
import os
import threading
import argparse
import numpy as np

from mindspore import Tensor, Parameter, save_checkpoint
from mindspore.train.serialization import load_checkpoint


def convert_ckpt(ckpt_path, save_path):
    checkpoint_dict = load_checkpoint(ckpt_path)
    checkpoint_dict = transpose_moe_gmm_checkpoint(checkpoint_dict)
    save_checkpoint(checkpoint_dict, save_path)


def transpose_moe_gmm_checkpoint(checkpoint_dict):
    for k, v in checkpoint_dict.items():
        if 'feed_forward.ffn.w1.weight' in k or \
            'feed_forward.ffn.w2.weight' in k or \
            'feed_forward.ffn.w3.weight' in k:
            checkpoint_dict[k] = Parameter(Tensor(np.transpose(v.asnumpy().astype(np.float32), (0, 2, 1))))
    return checkpoint_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='transpose gmm weight')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    threads = []
    for i in range(args.rank):
        src_ckpt_path = os.path.join(args.ckpt_path, f'rank_{i}/checkpoint_{i}.ckpt')
        dst_ckpt_path = os.path.join(args.save_path, f'rank_{i}/checkpoint_{i}.ckpt')
        os.makedirs(os.path.dirname(dst_ckpt_path), exist_ok=True)
        thread = threading.Thread(target=convert_ckpt, args=(src_ckpt_path, dst_ckpt_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
