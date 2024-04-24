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


def convert_hf_ckpt(ckpt_dir, output_name, dtype=ms.float16):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{ckpt_dir}'.", flush=True)
    try:
        ckpt_paths = sorted(Path(ckpt_dir).glob("*.safetensors"))
        dict_all = {}
        for i in range(len(ckpt_paths)):
            state_dict = load_file(ckpt_paths[i], device='cpu')
            dict_all.update(state_dict)
        model_hf = dict(sorted(dict_all.items(), key=lambda x: x[0]))
    # pylint: disable=W0703
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{ckpt_dir}', Error {e.message}.", flush=True)
        return False
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
                    ckpt_list.append({'name': name_w1, 'data': ms.Tensor(w1_value, dtype=dtype)})
                    w2_value = torch.stack(list_w2, 0).to(torch.float32).numpy()
                    print(f'\rprocessing parameter: {name_w2} {w2_value.shape}     ', end='', flush=True)
                    ckpt_list.append({'name': name_w2, 'data': ms.Tensor(w2_value, dtype=dtype)})
                    w3_value = torch.stack(list_w3, 0).to(torch.float32).numpy()
                    print(f'\rprocessing parameter: {name_w3} {w3_value.shape}     ', end='', flush=True)
                    ckpt_list.append({'name': name_w3, 'data': ms.Tensor(w3_value, dtype=dtype)})
                    count = 0
                    list_w1 = []
                    list_w2 = []
                    list_w3 = []

    ms.save_checkpoint(ckpt_list, output_name)
    print(f"\rConvert finished, the mindspore ckpt is saved in '{output_name}'.", flush=True)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_dir', default='./mixtral/torch_ckpt/')
    parser.add_argument('--mindspore_ckpt_path', default='./mixtral/ms_ckpt/')
    args = parser.parse_args()
    convert_hf_ckpt(ckpt_dir=args.torch_ckpt_dir, output_name=args.mindspore_ckpt_path)
