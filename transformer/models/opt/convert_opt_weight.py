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
from utils import print_state_dict, generate_total_layers_params


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
        if 'fc' in tgt and 'weight' in tgt or 'self_attn.out_proj.weight' in tgt:
            value = np.transpose(value, [1, 0])
        print(f"Mapping table Mindspore:{src:<30} \t Torch:{tgt:<30} with shape {value.shape}")
        new_ckpt_list.append({"data": Tensor(value), "name": src})
    return new_ckpt_list


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
                        default=None,
                        required=True,
                        help="The output mindspore checkpoint path.")

    opt = parser.parse_args()
    state_dict = torch.load(opt.torch_path, map_location='cpu')
    print_state_dict(state_dict)

    ms_name = [
        "backbone.transformer.encoder.blocks.{}.layernorm1.gama",
        "backbone.transformer.encoder.blocks.{}.layernorm1.beta",
        "backbone.transformer.encoder.blocks.{}.layernorm2.gama",
        "backbone.transformer.encoder.blocks.{}.layernorm2.beta",
        "backbone.transformer.encoder.blocks.{}.attention.projection.weight",
        "backbone.transformer.encoder.blocks.{}.attention.projection.beta",
        "backbone.transformer.encoder.blocks.{}.attention.dense1.weight",
        "backbone.transformer.encoder.blocks.{}.attention.dense1.bias",
        "backbone.transformer.encoder.blocks.{}.attention.dense2.weight",
        "backbone.transformer.encoder.blocks.{}.attention.dense2.bias",
        "backbone.transformer.encoder.blocks.{}.attention.dense3.weight",
        "backbone.transformer.encoder.blocks.{}.attention.dense3.bias",
        "backbone.transformer.encoder.blocks.{}.output.mapping.weight",
        "backbone.transformer.encoder.blocks.{}.output.mapping.bias",
        "backbone.transformer.encoder.blocks.{}.output.projection.weight",
        "backbone.transformer.encoder.blocks.{}.output.projection.bias"]

    torch_name = [
        "model.decoder.layers.{}.self_attn_layer_norm.weight",
        "model.decoder.layers.{}.self_attn_layer_norm.bias",
        "model.decoder.layers.{}.final_layer_norm.weight",
        "model.decoder.layers.{}.final_layer_norm.bias",

        "model.decoder.layers.{}.self_attn.out_proj.weight",
        "model.decoder.layers.{}.self_attn.out_proj.bias",
        "model.decoder.layers.{}.self_attn.q_proj.weight",
        "model.decoder.layers.{}.self_attn.q_proj.bias",
        "model.decoder.layers.{}.self_attn.k_proj.weight",
        "model.decoder.layers.{}.self_attn.k_proj.bias",
        "model.decoder.layers.{}.self_attn.v_proj.weight",
        "model.decoder.layers.{}.self_attn.v_proj.bias",

        "model.decoder.layers.{}.fc1.weight",
        "model.decoder.layers.{}.fc1.bias",
        "model.decoder.layers.{}.fc2.weight",
        "model.decoder.layers.{}.fc2.bias"
    ]

    addition_mindspore = [
        "head.lm_head.weight",
        "backbone.word_embedding.embedding_table",
        "backbone.position_embedding.embedding_table",
        "backbone.layernorm.gamma",
        "backbone.layernorm.beta"
    ]

    addition_torch = [
        "lm_head.weight",
        "model.decoder.embed_tokens.weight",
        "model.decoder.embed_positions.weight",
        "model.decoder.final_layer_norm.weight",
        "model.decoder.final_layer_norm.bias"
    ]

    mapped_param = generate_total_layers_params(total_layers=opt.layers,
                                                mindspore_params_per_layer=ms_name,
                                                torch_params_per_layer=torch_name,
                                                mindspore_additional_params=addition_mindspore,
                                                torch_additional_params=addition_torch)
    new_ckpt = get_converted_ckpt(mapped_param, state_dict)
    save_checkpoint(new_ckpt, opt.mindspore_path)
    print(f"Convert finished, the output is saved to {opt.mindspore_path}")
