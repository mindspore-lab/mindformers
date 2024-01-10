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

"""Convert MindSpore checkpoint to Torch"""
import os
import re
import argparse
import torch
from mindspore import load_checkpoint


def layer_name_mapping(key):
    """Convert huggingface PP weights mapping in MindSpore.

    return: new_name
    """
    prefix = ''
    # Handle first and last layers
    layer_rename_map = {
        "model.tok_embeddings.embedding_weight": "word_embeddings.weight",
        "attention_norm.weight": "input_layernorm.weight",
        "attention.wo.weight": "self_attention.dense.weight",
        "attention.wo.bias": "self_attention.dense.bias",
        "attention.wq.weight": "self_attention.query.weight",
        "attention.wk_v.weight": "self_attention.key_value.weight",
        "feed_forward.w1.weight": "mlp.gate_proj.weight",
        "feed_forward.w2.weight": "mlp.down_proj.weight",
        "feed_forward.w2.bias": "mlp.down_proj.bias",
        "feed_forward.w3.weight": "mlp.up_proj.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "model.norm_out.weight": "ln_f.weight"
    }
    if key in layer_rename_map:
        return prefix + layer_rename_map[key]

    match = re.compile(r'\w+\.\w+.(\d+)\.(.*)')
    layer_number = match.findall(key)[0][0]
    text = match.findall(key)[0][1]
    # Handle transformer blocks
    return f"{prefix}h.{layer_number}." + layer_rename_map[text]

def ms_to_torch(ms_weights):
    """Convert ms layers to torch."""
    torch_params = {}
    for k, v in ms_weights.items():
        new_name = layer_name_mapping(k)
        torch_params[new_name] = torch.from_numpy(v.asnumpy())
    return torch_params

def process_hf_shard_files(args):
    if args.torch_path and not os.path.exists(args.torch_path):
        os.makedirs(args.torch_path, exist_ok=True)

    file_name = "torch"
    ms_params = load_checkpoint(args.mindspore_path)
    torch_params = ms_to_torch(ms_params)
    save_file = args.torch_path + '/' + file_name + '.pth'
    torch.save(torch_params, save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Telechat convert script")
    parser.add_argument("--mindspore_path",
                        type=str,
                        default="",
                        help="The output mindspore checkpoint path.")
    parser.add_argument("--torch_path",
                        type=str,
                        default="",
                        help="The input torch checkpoint path.")
    config = parser.parse_args()

    # convert hf ckpt to ms
    process_hf_shard_files(args=config)
    current_path = os.getcwd()
    torch_ckpt_path = os.path.join(current_path, config.torch_path)
    print("*** finish ms convert torch model, torch_ckpt save in {} ***".format(torch_ckpt_path))
