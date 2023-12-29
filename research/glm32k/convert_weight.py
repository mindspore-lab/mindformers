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
Convert ChatGLM3-32K weight.
Support huggingface format.
"""
import argparse
import torch
from tqdm import tqdm
import mindspore as ms
from transformers import AutoModel


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


def merge_torch_model(args):
    """
        Merge all pytorch model files to a single mode

        Args:
            args: input paraameters.

        Returns:
            None
        """

    model = AutoModel.from_pretrained(args.huggingface_torch_path, trust_remote_code=True)

    with open("pt_model_arch.txt", "w") as fp:
        print(model, file=fp, flush=True)
    with open("pt_ckpt.txt", "w") as fp:
        for name, param in model.named_parameters():
            fp.write(f"{name} {param.shape} {param.dtype}\n")
    torch.save(model.state_dict(), args.merged_torch_path)


def get_converted_ckpt(pt_param, mindspore_path):
    """
    Print the keys of the loaded checkpoint

    Args:
        pt_param(dict): The loaded pytorch checkpoint.
        mindspore_path(str): the saved mindspore path

    Returns:
        None
    """
    ms_param = []
    with open("check_pt_ckpt.txt", "w") as fp:
        for k, v in tqdm(pt_param.items()):
            if "word_embeddings.weight" in k:
                k = k.replace("word_embeddings.weight", "embedding_table")
            fp.write(f"{k} {v.shape} {v.dtype}\n")
            ms_param.append({"name": k, "data": ms.Tensor(v.numpy(), dtype=ms.float16)})

    ms.save_checkpoint(ms_param, mindspore_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="chatGLM3-32k convert script")
    parser.add_argument('--layers',
                        type=int,
                        default=28,
                        help="The number of layers of the model to be converted.")
    parser.add_argument("--huggingface_torch_path",
                        type=str,
                        default="/path/pytorch_models/",
                        help="The original huggingface torch checkpoint path.")
    parser.add_argument("--merged_torch_path",
                        type=str,
                        default="/path/pytorch_models/glm32k.pth",
                        help="The merged torch checkpoint path.")
    parser.add_argument("--mindspore_path",
                        type=str,
                        default="/path/mindspore_models/glm32k.ckpt",
                        help="The mindspore modes")

    opt = parser.parse_args()
    merge_torch_model(opt)
    state_dict = torch.load(opt.merged_torch_path, map_location='cpu')
    print_dict(state_dict)
    get_converted_ckpt(state_dict, opt.mindspore_path)
    print(f"Convert finished, the output is saved to {opt.mindspore_path}")
