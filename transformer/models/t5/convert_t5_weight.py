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
    # Currently, the ms_extend_param the torch_extend_param is the full parameters.
    for src, tgt in mapped_params:
        value = weight_dict[tgt]
        if '.o.' in tgt and '.wi.' in tgt or '.wo.' in tgt:
            value = np.transpose(value, [1, 0])
        print(f"Mapping table Mindspore:{src:<30} \t Torch:{tgt:<30} with shape {value.shape}")
        value = weight_dict[tgt].numpy()
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
                        required=True,
                        default="The output mindspore checkpoint path.",
                        help="Use device nums, default is 128.")

    opt = parser.parse_args()
    state_dict = torch.load(opt.torch_path, map_location='cpu')
    print_state_dict(state_dict)

    ms_name = [
        "tfm_encoder.blocks.{}.layernorm1.gama",
        "tfm_encoder.blocks.{}.layernorm2.gama",
        "tfm_encoder.blocks.{}.attention.dense1.weight",
        "tfm_encoder.blocks.{}.attention.dense2.weight",
        "tfm_encoder.blocks.{}.attention.dense3.weight",
        "tfm_encoder.blocks.{}.attention.projection.weight",
        "tfm_encoder.blocks.{}.attention.bias_generator.embedding_table",
        "tfm_encoder.blocks.{}.output.mapping.weight",
        "tfm_encoder.blocks.{}.output.projection.weight",

        "tfm_decoder.blocks.{}.layernorm1.gama",
        "tfm_decoder.blocks.{}.cross_attention_layernorm.gama",
        "tfm_decoder.blocks.{}.layernorm2.gama",
        "tfm_decoder.blocks.{}.attention.dense1.weight",
        "tfm_decoder.blocks.{}.attention.dense2.weight",
        "tfm_decoder.blocks.{}.attention.dense3.weight",
        "tfm_decoder.blocks.{}.attention.projection.weight",
        "tfm_decoder.blocks.{}.attention.bias_generator.embedding_table",

        "tfm_decoder.blocks.{}.cross_attention.dense1.weight",
        "tfm_decoder.blocks.{}.cross_attention.dense2.weight",
        "tfm_decoder.blocks.{}.cross_attention.dense3.weight",
        "tfm_decoder.blocks.{}.cross_attention.projection.weight",
        "tfm_decoder.blocks.{}.attention.bias_generator.embedding_table",
        "tfm_decoder.blocks.{}.output.mapping.weight",
        "tfm_decoder.blocks.{}.output.projection.weight",
    ]

    torch_name = [
        "encoder.block.{}.layer.0.layer_norm.weight",
        "encoder.block.{}.layer.1.layer_norm.weight",
        "encoder.block.{}.layer.0.SelfAttention.q.weight",
        "encoder.block.{}.layer.0.SelfAttention.k.weight",
        "encoder.block.{}.layer.0.SelfAttention.v.weight",
        "encoder.block.{}.layer.0.SelfAttention.o.weight",
        "encoder.block.{}.layer.0.SelfAttention.relative_attention_bias.weight",

        "encoder.block.{}.layer.1.DenseReluDense.wi.weight",
        "encoder.block.{}.layer.1.DenseReluDense.wo.weight",

        "decoder.block.{}.layer.0.layer_norm.weight",
        "decoder.block.{}.layer.1.layer_norm.weight",
        "decoder.block.{}.layer.2.layer_norm.weight",
        "decoder.block.{}.layer.0.SelfAttention.q.weight",
        "decoder.block.{}.layer.0.SelfAttention.k.weight",
        "decoder.block.{}.layer.0.SelfAttention.v.weight",
        "decoder.block.{}.layer.0.SelfAttention.o.weight",
        "decoder.block.{}.layer.0.SelfAttention.relative_attention_bias.weight",

        "decoder.block.{}.layer.1.EncDecAttention.q.weight",
        "decoder.block.{}.layer.1.EncDecAttention.k.weight",
        "decoder.block.{}.layer.1.EncDecAttention.v.weight",
        "decoder.block.{}.layer.1.EncDecAttention.o.weight",
        "decoder.block.{}.layer.1.EncDecAttention.relative_attention_bias.weight",

        "decoder.block.{}.layer.1.DenseReluDense.wi.weight",
        "decoder.block.{}.layer.1.DenseReluDense.wo.weight",
    ]

    addition_mindspore = [
        "encoder_layernorm.gamma",
        "decoder_layernorm.gamma",
        "tfm_embedding_lookup.embedding_table"
    ]

    addition_torch = [
        "encoder_layer_norm.weight",
        "decoder_layer_norm.weight",
        "shared.weight"
    ]

    mapped_param = generate_total_layers_params(total_layers=opt.layers,
                                                mindspore_params_per_layer=ms_name,
                                                torch_params_per_layer=torch_name,
                                                mindspore_additional_params=addition_mindspore,
                                                torch_additional_params=addition_torch)
    new_ckpt = get_converted_ckpt(mapped_param, state_dict)
    save_checkpoint(new_ckpt, opt.mindspore_path)
    print(f"Convert finished, the output is saved to {opt.mindspore_path}")
