# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Convert checkpoint from torch/huggingface"""
import argparse
import numpy as np
import torch
from mindspore import save_checkpoint, Tensor
from tools.utils import print_state_dict, generate_total_layers_params

def get_converted_ckpt(mapped_params, weight_dict):
    """
    Print the keys of the loaded checkpoint

    Args:
        mapped_params(dict): The loaded checkpoint. The key is parameter name and value is the numpy array.
        weight_dict(dict): The loaded pytorch checkpoint.

    Returns:
        None
    """
    new_ckpt_list = []
    print(weight_dict.keys())
    # Currently, the ms_extend_param the torch_extend_param is the full parameters.
    for src, tgt in mapped_params:

        value = weight_dict[tgt].numpy()
        if '.dense.weight' in tgt:
            # print("tgt:", tgt)

            value = np.transpose(value, [1, 0])

        print(f"Mapping table Mindspore:{src:<30} \t Torch:{tgt:<30} with shape {value.shape}")

        new_ckpt_list.append({"data": Tensor(value), "name": src})
    return new_ckpt_list

def split_torch_attention(state):
    s = list(state.keys())
    for name in s:
        if name.endswith('attn.c_attn.weight') or name.endswith('attn.c_attn.bias'):
            value = state.pop(name)
            q, k, v = np.split(value.numpy(), 3, -1)
            state[name + '.q'] = torch.tensor(q, dtype=value.dtype)
            state[name + '.k'] = torch.tensor(k)
            state[name + '.v'] = torch.tensor(v)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OPT convert script")
    parser.add_argument('--layers',
                        type=int,
                        default=1,
                        help="The number of layers of the model to be converted.")
    parser.add_argument("--torch_path",
                        type=str,
                        default=None,
                        required=True,
                        help="The torch checkpoint path.")
    parser.add_argument("--mindspore_path",
                        type=str,
                        required=True,
                        default="The output mindspore checkpoint path.",
                        help="Use device nums, default is 128.")

    opt = parser.parse_args()
    state_dict = torch.load(opt.torch_path, map_location='cpu')
    print_state_dict(state_dict)

    ms_name = [
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.attention.dense1.weight",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.attention.dense1.bias",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.attention.dense2.weight",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.attention.dense2.bias",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.attention.dense3.weight",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.attention.dense3.bias",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.attention.projection.weight",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.attention.projection.bias",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.layernorm2.gamma",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.layernorm2.beta",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.output.mapping.weight",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.output.mapping.bias",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.output.projection.weight",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.output.projection.bias",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.layernorm1.gamma",
        "nezha.nezha.nezha_encoder.encoder.blocks.{}.layernorm1.beta",
    ]

    torch_name = [
        "nezha.encoder.layer.{}.attention.self.query.weight",
        "nezha.encoder.layer.{}.attention.self.query.bias",
        "nezha.encoder.layer.{}.attention.self.key.weight",
        "nezha.encoder.layer.{}.attention.self.key.bias",
        "nezha.encoder.layer.{}.attention.self.value.weight",
        "nezha.encoder.layer.{}.attention.self.value.bias",
        "nezha.encoder.layer.{}.attention.output.dense.weight",
        "nezha.encoder.layer.{}.attention.output.dense.bias",
        "nezha.encoder.layer.{}.attention.output.LayerNorm.weight",
        "nezha.encoder.layer.{}.attention.output.LayerNorm.bias",
        "nezha.encoder.layer.{}.intermediate.dense.weight",
        "nezha.encoder.layer.{}.intermediate.dense.bias",
        "nezha.encoder.layer.{}.output.dense.weight",
        "nezha.encoder.layer.{}.output.dense.bias",
        "nezha.encoder.layer.{}.output.LayerNorm.weight",
        "nezha.encoder.layer.{}.output.LayerNorm.bias",
    ]

    addition_mindspore = [
        "nezha.nezha.word_embedding.embedding_table",
        "nezha.nezha.embedding_postprocessor.token_type_embedding.embedding_table",
        "nezha.nezha.embedding_postprocessor.layernorm.gamma",
        "nezha.nezha.embedding_postprocessor.layernorm.beta",
        "nezha.nezha.dense.weight",
        "nezha.nezha.dense.bias",
    ]

    addition_torch = [
        "nezha.embeddings.word_embeddings.weight",
        "nezha.embeddings.token_type_embeddings.weight",
        "nezha.embeddings.LayerNorm.weight",
        "nezha.embeddings.LayerNorm.bias",
        "nezha.pooler.dense.weight",
        "nezha.pooler.dense.bias",
    ]

    mapped_param = generate_total_layers_params(total_layers=opt.layers,
                                                mindspore_params_per_layer=ms_name,
                                                torch_params_per_layer=torch_name,
                                                mindspore_additional_params=addition_mindspore,
                                                torch_additional_params=addition_torch)
    split_torch_attention(state_dict)
    new_ckpt = get_converted_ckpt(mapped_param, state_dict)
    save_checkpoint(new_ckpt, opt.mindspore_path)
    print(f"Convert finished, the output is saved to {opt.mindspore_path}")
