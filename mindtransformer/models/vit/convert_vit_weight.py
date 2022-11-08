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
from mindtransformer.utils import print_dict, generate_params_dict


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
        print(f"Mapping table Mindspore:{src:<30} \t Torch:{tgt:<30} with shape {value.shape}")
        value = weight_dict[tgt].numpy()
        if "mapping.weight" in src or "projection.weight" in src:
            value = np.transpose(value, [1, 0])
        new_ckpt_list.append({"data": Tensor(value), "name": src})
    return new_ckpt_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ViT convert script")
    parser.add_argument('--backbone_name',
                        type=str,
                        default='vit_base',
                        help="The network should be 'vit_base' or 'vit_large'")
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
    if opt.backbone_name == 'vit_base':
        opt.layers = 12
    elif opt.backbone_name == 'vit_large':
        opt.layers = 24
    else:
        raise ValueError(f"The network should be 'vit_base' or 'vit_large', but receive: {opt.backbone_name}")

    state_dict = torch.load(opt.torch_path, map_location='cpu')
    print_dict(state_dict)

    ms_name = [
        "body.transformer.encoder.blocks.{}.layernorm1.gamma",
        "body.transformer.encoder.blocks.{}.layernorm1.beta",
        "body.transformer.encoder.blocks.{}.attention.projection.weight",
        "body.transformer.encoder.blocks.{}.attention.projection.bias",
        "body.transformer.encoder.blocks.{}.attention.dense1.weight",
        "body.transformer.encoder.blocks.{}.attention.dense1.bias",
        "body.transformer.encoder.blocks.{}.attention.dense2.weight",
        "body.transformer.encoder.blocks.{}.attention.dense2.bias",
        "body.transformer.encoder.blocks.{}.attention.dense3.weight",
        "body.transformer.encoder.blocks.{}.attention.dense3.bias",
        "body.transformer.encoder.blocks.{}.layernorm2.gamma",
        "body.transformer.encoder.blocks.{}.layernorm2.beta",
        "body.transformer.encoder.blocks.{}.output.mapping.weight",
        "body.transformer.encoder.blocks.{}.output.mapping.bias",
        "body.transformer.encoder.blocks.{}.output.projection.weight",
        "body.transformer.encoder.blocks.{}.output.projection.bias",
    ]

    torch_name = [
        "vit.encoder.layer.{}.layernorm_before.weight",
        "vit.encoder.layer.{}.layernorm_before.bias",
        "vit.encoder.layer.{}.attention.output.dense.weight",
        "vit.encoder.layer.{}.attention.output.dense.bias",
        "vit.encoder.layer.{}.attention.attention.query.weight",
        "vit.encoder.layer.{}.attention.attention.query.bias",
        "vit.encoder.layer.{}.attention.attention.key.weight",
        "vit.encoder.layer.{}.attention.attention.key.bias",
        "vit.encoder.layer.{}.attention.attention.value.weight",
        "vit.encoder.layer.{}.attention.attention.value.bias",
        "vit.encoder.layer.{}.layernorm_after.weight",
        "vit.encoder.layer.{}.layernorm_after.bias",
        "vit.encoder.layer.{}.intermediate.dense.weight",
        "vit.encoder.layer.{}.intermediate.dense.bias",
        "vit.encoder.layer.{}.output.dense.weight",
        "vit.encoder.layer.{}.output.dense.bias",
    ]

    addition_mindspore = [
        "cls",
        "stem.patch_to_embedding.weight",
        "stem.patch_to_embedding.bias",
        "pos_embedding",
        "norm.gamma",
        "norm.beta",
        "head.0.weight",
        "head.0.bias",
    ]

    addition_torch = [
        "vit.embeddings.cls_token",
        "vit.embeddings.patch_embeddings.projection.weight",
        "vit.embeddings.patch_embeddings.projection.bias",
        "vit.embeddings.position_embeddings",
        "vit.layernorm.weight",
        "vit.layernorm.bias",
        "classifier.weight",
        "classifier.bias",
    ]

    mapped_param = generate_params_dict(total_layers=opt.layers,
                                        mindspore_params_per_layer=ms_name,
                                        torch_params_per_layer=torch_name,
                                        mindspore_additional_params=addition_mindspore,
                                        torch_additional_params=addition_torch)
    new_ckpt = get_converted_ckpt(mapped_param, state_dict)
    save_checkpoint(new_ckpt, opt.mindspore_path)
    print(f"Convert finished, the output is saved to {opt.mindspore_path}")
