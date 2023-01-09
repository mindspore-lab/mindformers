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
import mindspore as ms
import torch


def convert_weight(pth_file="mae_pretrain_vit_base.pth", ms_ckpt_path="mae_pretrain_vit_base.ckpt"):
    """
    convert mae_vit_base_p16 weights from pytorch to mindspore
    pytorch and GPU required.
    """
    ms_ckpt = []
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    for k, v in state_dict['model'].items():
        if k in ("cls_token", "mask_token"):
            k += "s"
        if "norm" in k:
            if "fc_norm" not in k:
                if "blocks" in k:
                    k = k.replace("norm", "layernorm")
                else:
                    k = k.replace("norm", "fc_norm")
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
        if 'qkv' not in k:
            ms_ckpt.append({'name': k, 'data': ms.Tensor(v.numpy())})
            print(k)
        else:
            if k.startswith("decoder"):
                data = ms.Tensor(v.numpy())
                ms_ckpt.append({'name': k.replace('.qkv', '.dense1'), 'data': data[:512]})
                ms_ckpt.append({'name': k.replace('.qkv', '.dense2'), 'data': data[512:1024]})
                ms_ckpt.append({'name': k.replace('.qkv', '.dense3'), 'data': data[1024:]})
                print(k.replace('.qkv', '.dense1'))
                print(k.replace('.qkv', '.dense2'))
                print(k.replace('.qkv', '.dense3'))
            else:
                data = ms.Tensor(v.numpy())
                ms_ckpt.append({'name': k.replace('.qkv', '.dense1'), 'data': data[:768]})
                ms_ckpt.append({'name': k.replace('.qkv', '.dense2'), 'data': data[768:1536]})
                ms_ckpt.append({'name': k.replace('.qkv', '.dense3'), 'data': data[1536:]})
                print(k.replace('.qkv', '.dense1'))
                print(k.replace('.qkv', '.dense2'))
                print(k.replace('.qkv', '.dense3'))

    ms.save_checkpoint(ms_ckpt, ms_ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mae vit weight convert script")
    parser.add_argument("--torch_path",
                        type=str,
                        default="mae_pretrain_vit_base.pth",
                        required=True,
                        help="The torch checkpoint path.")
    parser.add_argument("--mindspore_path",
                        type=str,
                        required=True,
                        default="mae_pretrain_vit_base.ckpt",
                        help="The output mindspore checkpoint path.")
    opt = parser.parse_args()

    convert_weight(opt.torch_path, opt.mindspore_path)
