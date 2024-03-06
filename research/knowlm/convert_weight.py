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
Convert llama weight.
Support huggingface format and Meta format.
"""

import os
import json
import argparse
import mindspore as ms

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
    name = name.replace('.self_attn.q_proj.', '.attention.wq.')
    name = name.replace('.self_attn.k_proj.', '.attention.wk.')
    name = name.replace('.self_attn.v_proj.', '.attention.wv.')
    name = name.replace('.self_attn.o_proj.', '.attention.wo.')
    name = name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    name = name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    name = name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    name = name.replace('.input_layernorm.', '.attention_norm.')
    name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
    name = name.replace('.norm.', '.norm_out.')
    return name

def convert_hf_ckpt(ckpt_dir, output_name, dtype=ms.float16):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{ckpt_dir}'.", flush=True)
    try:
        from transformers import LlamaForCausalLM

    except:
        raise ImportError(f"Failed to load huggingface checkpoint. Please make sure transformers is available.")

    try:
        model_hf = LlamaForCausalLM.from_pretrained(ckpt_dir)
        args_hf = read_json(os.path.join(ckpt_dir, "config.json"))
        print(args_hf)
    # pylint: disable=W0703
    except Exception as e:
        print(f"Error {e.message}.", flush=True)
        return False
    dim = args_hf["hidden_size"]
    ckpt_list = []
    for name, value in model_hf.named_parameters():
        name = name_replace(name)
        if 'W_pack' in name:
            values = torch.split(value, dim)
            wq = name.replace('.self_attn.W_pack', '.attention.wq') #'.self_attn.q_proj.', '.attention.wq.'
            q_value = values[0]
            wk = name.replace('.self_attn.W_pack', '.attention.wk')
            k_value = values[1]
            wv = name.replace('.self_attn.W_pack', '.attention.wv')
            v_value = values[2]
            print(f'\rprocessing parameter: {wq} {q_value.shape}     ', end='', flush=True)
            ckpt_list.append({'name': wq, 'data': ms.Tensor(q_value.detach().numpy(), dtype=dtype)})
            print(f'\rprocessing parameter: {wk} {k_value.shape}     ', end='', flush=True)
            ckpt_list.append({'name': wk, 'data': ms.Tensor(k_value.detach().numpy(), dtype=dtype)})
            print(f'\rprocessing parameter: {wv} {v_value.shape}     ', end='', flush=True)
            ckpt_list.append({'name': wv, 'data': ms.Tensor(v_value.detach().numpy(), dtype=dtype)})
            continue
        if name == 'norm.weight':
            name = 'norm_out.weight'
        if name[:7] == 'layers.':
            name = name[7:]
        value = value.detach().numpy()
        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': ms.Tensor(value, dtype=dtype)})
    ckpt_file = os.path.join(ckpt_dir, output_name)
    ms.save_checkpoint(ckpt_list, os.path.join(ckpt_file))
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{ckpt_file}'.", flush=True)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_dir', default='./llama_model/llama-13b-hf/')
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    args = parser.parse_args()
    convert_hf_ckpt(ckpt_dir=args.torch_ckpt_dir, output_name=args.mindspore_ckpt_path)
