# Copyright 2023 Huawei Technologies Co., Ltd
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
"""transform ckpt"""
import os
import argparse
import mindspore as ms


def checkpoint_qkv_ffn_weight_name(src_ckpt_dir):
    """check ckpt contain qkv"""
    src_ckpt = os.path.join(src_ckpt_dir, "rank_0")
    for name in os.listdir(src_ckpt):
        if name.endswith('.ckpt'):
            checkpoint_path = os.path.join(src_ckpt, name)
            param_dict = ms.load_checkpoint(checkpoint_path)
            for i in param_dict.keys():
                if "w_qkv" in i:
                    return 'yes-qkv'
                if "w_gate_hidden" in i:
                    return 'yes-qkv'
    return 'no-qkv'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt_dir',
                        default="",
                        type=str,
                        help='path of src ckpt')
    args = parser.parse_args()
    result = checkpoint_qkv_ffn_weight_name(args.src_ckpt_dir)
    #为什么不返回true/false: 有个stdcout打印里有kbk_cache:False/True会混淆
    print(result)
