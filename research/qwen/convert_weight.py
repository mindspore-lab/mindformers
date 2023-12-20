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
"""
Convert Qwen weight.
Support huggingface format.
"""

import argparse
import os

import mindspore as ms
import numpy as np
import torch
from transformers import AutoModelForCausalLM

ATTENTION_WEIGHT_NAME = 'attn.c_attn.weight'
ATTENTION_BIAS_NAME = 'attn.c_attn.bias'


def _name_replace(name: str):
    """replace huggingface parameter name to mindformers."""
    name = name.replace('.h.', '.layers.')

    name = name.replace('.wte.weight', '.wte.embedding_weight')

    name = name.replace('attn.c_proj.', "attention.wo.")

    name = name.replace('ln_1.', 'attention_norm.')
    name = name.replace('ln_2.', 'ffn_norm.')

    name = name.replace('mlp.w1.', 'feed_forward.w1.')
    name = name.replace('mlp.w2.', 'feed_forward.w3.')
    name = name.replace('mlp.c_proj.', 'feed_forward.w2.')
    return name


def convert_attention_weight(name, value, ckpt_weights, dtype=ms.float16):
    split_arr = np.array_split(value, 3)
    attention_weight_names = ['attention.wq.weight', 'attention.wk.weight', 'attention.wv.weight']

    for index in range(len(split_arr)):
        cur_name = name.replace(ATTENTION_WEIGHT_NAME, attention_weight_names[index])
        ckpt_weights.append({'name': cur_name, 'data': ms.Tensor(split_arr[index], dtype=dtype)})


def convert_attention_bias(name, value, ckpt_weights, dtype=ms.float16):
    split_arr = np.array_split(value, 3)
    attention_bias_names = ['attention.wq.bias', 'attention.wk.bias', 'attention.wv.bias']

    for index in range(len(split_arr)):
        cur_name = name.replace(ATTENTION_BIAS_NAME, attention_bias_names[index])
        ckpt_weights.append({'name': cur_name, 'data': ms.Tensor(split_arr[index], dtype=dtype)})


def convert_hf_ckpt(ckpt_dir, output_file_path, torch_dtype=torch.float16, dtype=ms.float16):
    """convert huggingface weights files to mindspore."""
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map="cpu", trust_remote_code=True)

    ckpt_weights = []
    for k, v in model.named_parameters():
        print('Parameter (name=%s, shape=%s, dtype=%s, requires_grad=%s)' % (k, v.shape, v.dtype, v.requires_grad))
        try:
            value = v.detach().numpy()
        except TypeError:
            value = v.detach().to(torch_dtype).cpu().numpy()
            print('dtype:  %s->%s' % (v.dtype, value.dtype))

        msname = _name_replace(k)
        if msname != k:
            print('name:  %s->%s' % (k, msname))

        if ATTENTION_WEIGHT_NAME in msname:
            convert_attention_weight(msname, value, ckpt_weights, dtype)
            continue

        if ATTENTION_BIAS_NAME in msname:
            convert_attention_bias(msname, value, ckpt_weights, dtype)
            continue

        ckpt_weights.append({'name': msname, 'data': ms.Tensor(value, dtype=dtype)})

    print('Saving converted weights to %s...' % output_file_path)
    ms.save_checkpoint(ckpt_weights, output_file_path)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Qwen convert script")
    parser.add_argument("--torch_ckpt_dir",
                        required=True,
                        help="The torch checkpoint path.")
    parser.add_argument("--mindspore_ckpt_path",
                        default="./run/qwen_7b_ms.ckpt",
                        help="The output checkpoint path.")

    args = parser.parse_args()

    output_path = os.path.expanduser(args.mindspore_ckpt_path)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    convert_hf_ckpt(args.torch_ckpt_dir, output_path, torch_dtype=torch.float32, dtype=ms.float32)
