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
import mindspore as ms
import torch as pt


def convert_pt_to_ms(pt_pth_path, ms_ckpt_path):
    """ Convert pytorch mpdel file to MindSpore model file. """
    pt_param = pt.load(pt_pth_path)

    print('parameter convert....')
    ms_param = []
    for k, v in pt_param.items():
        if "word_embeddings.weight" in k:
            k = k.replace("weight", "embedding_table")
        if "post_attention_layernorm" in k or "input_layernorm" in k or "final_layernorm" in k:
            k = k.replace("weight", "gamma")
            k = k.replace("bias", "beta")
        if "input_layernorm" in k:
            v = v.half()
        ms_param.append({"name": k, "data": ms.Tensor(v.numpy())})

    print('saving ms ckpt....')
    ms.save_checkpoint(ms_param, ms_ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ChatGLM6B weight convert script")
    parser.add_argument("--pt_ckpt_path",
                        type=str,
                        required=True,
                        default="None",
                        help="The torch checkpoint path.")
    parser.add_argument("--ms_ckpt_path",
                        type=str,
                        required=False,
                        default="ms_glm_6b.ckpt",
                        help='The output mindspore checkpoint path.')

    opt = parser.parse_args()
    convert_pt_to_ms(opt.pt_ckpt_path, opt.ms_ckpt_path)
    print(f"Convert finished, the output is saved to {opt.ms_ckpt_path}")
