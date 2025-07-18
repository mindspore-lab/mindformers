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
"""
Convert mixtral weight.
Support huggingface format
"""

import argparse
from pathlib import Path
import torch
import mindspore as ms
from safetensors.torch import load_file
from mindformers.tools.utils import str2bool

dtype_map = {
    'fp32': ms.float32,
    'bf16': ms.bfloat16,
    'fp16': ms.float16
}


def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
    name = name.replace('.block_sparse_moe.gate.', '.feed_forward.router.dense.')
    name = name.replace('.block_sparse_moe.experts.', '.feed_forward.ffn.')
    name = name.replace('.self_attn.k_proj.', '.attention.wk.')
    name = name.replace('.self_attn.o_proj.', '.attention.wo.')
    name = name.replace('.self_attn.q_proj.', '.attention.wq.')
    name = name.replace('.self_attn.v_proj.', '.attention.wv.')
    name = name.replace('.input_layernorm.', '.attention_norm.')
    name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
    return name


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, use_gmm=False, **kwargs):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        ckpt_paths = sorted(Path(input_path).glob("*.safetensors"))
        dict_all = {}
        for i, _ in enumerate(ckpt_paths):
            state_dict = load_file(ckpt_paths[i], device='cpu')
            dict_all.update(state_dict)
        model_hf = dict(sorted(dict_all.items(), key=lambda x: x[0]))
    # pylint: disable=W0703
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{input_path}', Error {e.message}.", flush=True)
    ckpt_list = []
    count = 0
    list_w1 = []
    list_w2 = []
    list_w3 = []
    for name, value in model_hf.items():
        name = name_replace(name)
        if name == 'model.norm.weight':
            name = 'model.norm_out.weight'
            value = value.to(torch.float32).numpy()
            print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
            ckpt_list.append({'name': name, 'data': ms.Tensor(value, dtype=dtype)})
        else:
            if 'feed_forward.ffn' not in name:
                value = value.to(torch.float32).numpy()
                print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
                ckpt_list.append({'name': name, 'data': ms.Tensor(value, dtype=dtype)})
            else:
                # 3:number of linear(w1,w2,w3) ,24 = 3 * 8(number of linear * expert_num)
                if count % 3 == 0 and count != 24:
                    list_w1.append(value)
                if count % 3 == 1 and count != 24:
                    list_w2.append(value)
                if count % 3 == 2 and count != 24:
                    list_w3.append(value)
                count = count + 1
                if count == 24:
                    str_front = name.split('ffn')[0]
                    name_w1 = str_front + 'ffn.w1.weight'
                    name_w2 = str_front + 'ffn.w2.weight'
                    name_w3 = str_front + 'ffn.w3.weight'
                    w1_value = torch.stack(list_w1, 0).to(torch.float32).numpy()
                    print(f'\rprocessing parameter: {name_w1} {w1_value.shape}     ', end='', flush=True)
                    ckpt_list.append({'name': name_w1,
                                      'data': ms.Tensor(w1_value if not use_gmm else w1_value.transpose((0, 2, 1)),
                                                        dtype=dtype)})
                    w2_value = torch.stack(list_w2, 0).to(torch.float32).numpy()
                    print(f'\rprocessing parameter: {name_w2} {w2_value.shape}     ', end='', flush=True)
                    ckpt_list.append({'name': name_w2,
                                      'data': ms.Tensor(w2_value if not use_gmm else w2_value.transpose((0, 2, 1)),
                                                        dtype=dtype)})
                    w3_value = torch.stack(list_w3, 0).to(torch.float32).numpy()
                    print(f'\rprocessing parameter: {name_w3} {w3_value.shape}     ', end='', flush=True)
                    ckpt_list.append({'name': name_w3,
                                      'data': ms.Tensor(w3_value if not use_gmm else w3_value.transpose((0, 2, 1)),
                                                        dtype=dtype)})
                    count = 0
                    list_w1 = []
                    list_w2 = []
                    list_w3 = []

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert finished, the mindspore ckpt is saved in '{output_path}'.", flush=True)


def convert_ms_to_gmm(input_path, output_path, **kwargs):
    """convert ms weight to gmm."""
    params = ms.load_checkpoint(input_path)
    for k, v in params.items():
        if 'feed_forward.ffn.w1.weight' in k or \
                'feed_forward.ffn.w2.weight' in k or \
                'feed_forward.ffn.w3.weight' in k:
            orig_tensor = ms.Tensor(v)
            gmm_tensor = orig_tensor.transpose((0, 2, 1))
            params[k] = ms.Parameter(gmm_tensor)
            print(f"\rConvert {params[k]} to gmm weight.", flush=True)
    ms.save_checkpoint(params, output_path)
    print(f"\rConvert finished, the mindspore ckpt is saved in '{output_path}'.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_dir', default='./mixtral/torch_ckpt/')
    parser.add_argument('--mindspore_ckpt_path', default='./mixtral/ms_ckpt/')
    parser.add_argument('--use_gmm', type=str2bool, default=False)
    parser.add_argument('--pre_ckpt_path', default=None)
    parser.add_argument('--dtype', default='fp16', type=str, choices=['fp16', 'fp32', 'bf16'])
    args = parser.parse_args()

    if args.pre_ckpt_path:
        convert_ms_to_gmm(input_path=args.pre_ckpt_path, output_path=args.mindspore_ckpt_path)
    else:
        convert_pt_to_ms(input_path=args.torch_ckpt_dir, output_path=args.mindspore_ckpt_path,
                         dtype=dtype_map.get(args.dtype), use_gmm=args.use_gmm)
