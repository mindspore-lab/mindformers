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
"""Convert checkpoint from torch/facebook"""
import argparse
import torch
import mindspore as ms


def convert_weight(torch_pth_path="mae_pretrain_vit_base.pth", ms_ckpt_path="mae_pretrain_vit_base.ckpt"):
    """
    convert mae_vit_base_p16 weights from pytorch to mindspore
    pytorch and GPU required.
    """
    param_dict = torch.load(torch_pth_path, map_location=torch.device("cpu"))

    new_dict = []
    for k, v in param_dict["model"].items():
        if k in ("cls_token", "mask_token"):
            k += "s"
        if "head" not in k:
            k = "vit." + k
        if "norm" in k:
            if "fc_norm" not in k and "vit.norm" not in k and "decoder_norm" not in k:
                k = k.replace("norm", "layernorm")
            if "weight" in k:
                k = k.replace("weight", "gamma")
            elif "bias" in k:
                k = k.replace("bias", "beta")
        if "mlp" in k:
            k = k.replace("mlp", "output")
            if "fc1" in k:
                k = k.replace("fc1", "mapping")
                if "weight" in k:
                    v = v.transpose(-1, 0)
            elif "fc2" in k:
                k = k.replace("fc2", "projection")
                if "weight" in k:
                    v = v.transpose(-1, 0)
        if "attn" in k:
            k = k.replace("attn", "attention")
            if "proj" in k:
                k = k.replace("proj", "projection")
                if "weight" in k:
                    v = v.transpose(-1, 0)
        if "qkv" not in k:
            new_dict.append({"name": k, "data": ms.Tensor(v.numpy())})
        else:
            data = ms.Tensor(v.numpy())
            length = data.shape[0] // 3
            new_dict.append({"name": k.replace(".qkv", ".dense1"), "data": data[:length]})
            new_dict.append({"name": k.replace(".qkv", ".dense2"), "data": data[length:length*2]})
            new_dict.append({"name": k.replace(".qkv", ".dense3"), "data": data[length*2:]})

    ms.save_checkpoint(new_dict, ms_ckpt_path)

    print("Weights conversion completes. ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mae vit weight convert script")
    parser.add_argument("--torch_pth_path",
                        type=str,
                        default="mae_pretrain_vit_base.pth",
                        required=True,
                        help="The torch checkpoint path.")
    parser.add_argument("--ms_ckpt_path",
                        type=str,
                        required=True,
                        default="mae_pretrain_vit_base.ckpt",
                        help="The output mindspore checkpoint path.")
    opt = parser.parse_args()

    convert_weight(opt.torch_pth_path, opt.ms_ckpt_path)
