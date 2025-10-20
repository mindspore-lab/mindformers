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
"""convert weight."""
import argparse
import copy
import importlib

import torch
import mindspore as ms

from mindformers.tools.utils import str2bool

dtype_map = {
    'fp32': ms.float32,
    'bf16': ms.bfloat16,
    'fp16': ms.float16
}

reversed_dtype_map = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp16': torch.float16
}

convert_map = {
    'llama': 'mindformers.models.llama.convert_weight.convert_pt_to_ms',
    'qwen2_5': 'research.qwen2_5.convert_weight.convert_weight',
    'glm-n': 'mindformers.models.glm2.convert_weight.convert_weight',
    'mixtral': 'research.mixtral.convert_weight.convert_pt_to_ms',
    'telechat': 'research.telechat.convert_weight.convert_pt_to_ms',
    'deepseekv3': 'toolkit.weight_convert.deepseekv3.convert_deepseekv3_hf_weight.convert_weight'
}

reversed_convert_map = {
    'llama': 'mindformers.models.llama.convert_reversed.convert_ms_to_pt',
    'glm-n': 'mindformers.models.glm2.convert_reversed.convert_ms_to_pt',
    'mixtral': 'research.mixtral.convert_reversed.convert_ms_to_pt',
    'telechat': 'research.telechat.convert_reversed.convert_ms_to_pt',
    'deepseekv3': 'toolkit.weight_convert.deepseekv3.reverse_mcore_deepseekv3_weight_to_hf.reverse_weight',
    'qwen3': 'toolkit.weight_convert.qwen3.reverse_mcore_qwen3_weight_to_hf.reverse_weight',
    'qwen3-moe': 'toolkit.weight_convert.qwen3_moe.reverse_mcore_qwen3_moe_weight_to_hf.reverse_weight'
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, required=True, help='model name')
    parser.add_argument('--reversed', action='store_true', help="convert ms to hf")
    parser.add_argument('--input_path', default=None, type=str, required=True)
    parser.add_argument('--output_path', default=None, type=str, required=True)
    parser.add_argument('--dtype', default=None, type=str, required=False)
    parser.add_argument('--qkv_concat', default=False, type=str2bool, required=False)

    parser.add_argument('--telechat_type', default="telechat_12b", type=str, required=False,
                        help="Only for telechat. Telechat version.")
    parser.add_argument('--is_lora', default=False, type=str2bool, required=False)

    args, extra_args = parser.parse_known_args()
    extra_args = [i
                  for item in extra_args
                  for i in item.split("=")]

    extra_kwargs = copy.copy(vars(args))
    extra_kwargs.pop('model')
    extra_kwargs.pop('reversed')
    extra_kwargs.pop('input_path')
    extra_kwargs.pop('output_path')
    extra_kwargs.pop('dtype')
    while extra_args:
        key = extra_args.pop(0)
        value = extra_args.pop(0)
        if not key.startswith("--"):
            raise ValueError("Custom config key need to start with --.")
        extra_kwargs[key[2:]] = value

    if args.model in ["glm4"]:
        args.model = "glm-n"

    if args.reversed:
        module_func = reversed_convert_map.get(args.model)
        dtype = reversed_dtype_map.get(args.dtype)
    else:
        module_func = convert_map.get(args.model)
        dtype = dtype_map.get(args.dtype)

    if not module_func:
        raise ValueError(f"Model:{args.model} is not supported!\nSupported Models:{','.join(convert_map.keys())}.")
    if args.dtype and not dtype:
        raise ValueError(f"Dtype:{args.dtype} is not supported!\nSupported Models:{','.join(dtype_map.keys())}.\n")

    model_name, func_name = module_func.rsplit('.', 1)
    convert_func = getattr(importlib.import_module(model_name), func_name)

    merged_args = argparse.Namespace(**{**vars(args), **extra_kwargs})
    convert_func(merged_args)
