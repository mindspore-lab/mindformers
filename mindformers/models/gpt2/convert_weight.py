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
"""Convert checkpoint from torch/huggingface"""
import argparse
import numpy as np
import torch
from mindspore import save_checkpoint, Tensor

def generate_params_dict(total_layers,
                         mindspore_params_per_layer,
                         torch_params_per_layer,
                         mindspore_additional_params,
                         torch_additional_params):
    """
    Generate the total parameter mapping of mindspore and pytorch.

    Args:
        total_layers(int): The total layers of the net.
        mindspore_params_per_layer(list): The list of params per layer for the net of mindspore.
        torch_params_per_layer(list): The list of params per layer for the net of pytorch.
        mindspore_additional_params(list): The list of params outside the layer for the net of mindspore
        torch_additional_params(list): The list  of params outside the layer for the net of pytorch.

    Returns:
        A list of tuple. The first element is the parameter name of mindspore,
        the another is the parameter name of pytorch.
    """
    mapped_params = list(zip(mindspore_params_per_layer, torch_params_per_layer))
    ms_extend_param_list = []
    torch_extend_param_list = []
    for i in range(total_layers):
        for ms_para, torch_para in mapped_params:
            src = ms_para.format(i)
            tgt = torch_para.format(i)

            ms_extend_param_list.append(src)
            torch_extend_param_list.append(tgt)

    mapped_params = list(zip(mindspore_additional_params, torch_additional_params))
    for ms_para, torch_para in mapped_params:
        ms_extend_param_list.append(ms_para)
        torch_extend_param_list.append(torch_para)

    return list(zip(ms_extend_param_list, torch_extend_param_list))


def print_dict(input_dict):
    """
    Print the keys and values of input dict

    Args:
        input_dict(dict): input dict with key and value.

    Returns:
        None
    """
    for k, v in input_dict.items():
        print(f"Param: {k} with shape {v}")


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
    # Currently, the ms_extend_param the torch_extend_param is the full parameters.
    for src, tgt in mapped_params:
        value = weight_dict[tgt].numpy()
        # split the attention layer for q, k, v

        if 'c_attn.weight' in tgt:
            print("tgt:", tgt)
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
                        default=12,
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
    print_dict(state_dict)

    ms_name = [
        "gpt2.backbone.transformer.encoder.blocks.{}.layernorm1.gamma",
        "gpt2.backbone.transformer.encoder.blocks.{}.layernorm1.beta",
        "gpt2.backbone.transformer.encoder.blocks.{}.layernorm2.gamma",
        "gpt2.backbone.transformer.encoder.blocks.{}.layernorm2.beta",
        "gpt2.backbone.transformer.encoder.blocks.{}.attention.projection.weight",
        "gpt2.backbone.transformer.encoder.blocks.{}.attention.projection.bias",
        "gpt2.backbone.transformer.encoder.blocks.{}.attention.dense1.weight",
        "gpt2.backbone.transformer.encoder.blocks.{}.attention.dense1.bias",
        "gpt2.backbone.transformer.encoder.blocks.{}.attention.dense2.weight",
        "gpt2.backbone.transformer.encoder.blocks.{}.attention.dense2.bias",
        "gpt2.backbone.transformer.encoder.blocks.{}.attention.dense3.weight",
        "gpt2.backbone.transformer.encoder.blocks.{}.attention.dense3.bias",
        "gpt2.backbone.transformer.encoder.blocks.{}.output.mapping.weight",
        "gpt2.backbone.transformer.encoder.blocks.{}.output.mapping.bias",
        "gpt2.backbone.transformer.encoder.blocks.{}.output.projection.weight",
        "gpt2.backbone.transformer.encoder.blocks.{}.output.projection.bias",
    ]

    torch_name = [
        "transformer.h.{}.ln_1.weight",
        "transformer.h.{}.ln_1.bias",
        "transformer.h.{}.ln_2.weight",
        "transformer.h.{}.ln_2.bias",
        "transformer.h.{}.attn.c_proj.weight",
        "transformer.h.{}.attn.c_proj.bias",
        "transformer.h.{}.attn.c_attn.weight.q",
        "transformer.h.{}.attn.c_attn.bias.q",
        "transformer.h.{}.attn.c_attn.weight.k",
        "transformer.h.{}.attn.c_attn.bias.k",
        "transformer.h.{}.attn.c_attn.weight.v",
        "transformer.h.{}.attn.c_attn.bias.v",
        "transformer.h.{}.mlp.c_fc.weight",
        "transformer.h.{}.mlp.c_fc.bias",
        "transformer.h.{}.mlp.c_proj.weight",
        "transformer.h.{}.mlp.c_proj.bias"
    ]

    addition_mindspore = [
        "gpt2.backbone.layernorm.gamma",
        "gpt2.backbone.layernorm.beta",
        "backbone.embedding.word_embedding.embedding_table",
        "backbone.embedding.position_embedding.embedding_table",
        # "gpt2.dense1.weight",   # for the model with head
    ]

    addition_torch = [
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
        "transformer.wte.weight",
        "transformer.wpe.weight",
        # "transformer.wte.weight",   # for the model with head
    ]

    mapped_param = generate_params_dict(total_layers=opt.layers,
                                        mindspore_params_per_layer=ms_name,
                                        torch_params_per_layer=torch_name,
                                        mindspore_additional_params=addition_mindspore,
                                        torch_additional_params=addition_torch)
    split_torch_attention(state_dict)
    new_ckpt = get_converted_ckpt(mapped_param, state_dict)
    save_checkpoint(new_ckpt, opt.mindspore_path)
    print(f"Convert finished, the output is saved to {opt.mindspore_path}")
