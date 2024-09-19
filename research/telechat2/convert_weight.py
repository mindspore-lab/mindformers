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

"""
Convert Telechat weight.
Support huggingface format.
"""

import os
import json
import argparse

import mindspore as ms

from mindformers.tools import logger
from mindformers.utils.convert_utils import pt2ms


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('transformer.word_embeddings.weight', 'model.tok_embeddings.embedding_weight')
    name = name.replace('.input_layernorm', '.attention_norm')
    name = name.replace('.self_attention.dense.', '.attention.wo.')
    name = name.replace('.self_attention.dense.bias.', '.attention.wo.bias.')
    name = name.replace('.self_attention.query.', '.attention.wq.')
    name = name.replace('.self_attention.key_value.', '.attention.wk_v.')
    name = name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    name = name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    name = name.replace('.mlp.down_proj.bias.', '.feed_forward.w2.bias.')
    name = name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    name = name.replace('.post_attention_layernorm.', '.ffn_norm.weight.')
    name = name.replace('lm_head.', 'lm_head.')
    name = name.replace('transformer.ln_f.', 'model.norm_out.')
    return name


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert baichuan hf weight to ms."""
    ckpt_dir = os.path.dirname(input_path)
    print(f"Trying to convert huggingface checkpoint in '{ckpt_dir}'.", flush=True)
    try:
        from transformers import AutoModelForCausalLM
        model_hf = AutoModelForCausalLM.from_pretrained(ckpt_dir, trust_remote_code=True)
    # pylint: disable=W0703
    except Exception as e:
        print(f"Can not find huggingface checkpoint in '{ckpt_dir}', Error {e}.", flush=True)
        return False

    ckpt_list = []
    for name, value in model_hf.state_dict().items():
        name = name_replace(name)

        if name.startswith("h."):
            name = name.replace('h.', 'model.layers.')
        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    logger.info(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
                flush=True)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Telechat convert script")
    parser.add_argument("--torch_path",
                        type=str,
                        default="",
                        help="The input torch checkpoint path.")
    parser.add_argument("--mindspore_path",
                        type=str,
                        default="",
                        help="The output mindspore checkpoint path.")
    parser.add_argument("--dtype", default=None, choices=['float16', 'float32', 'bfloat16'],
                        help="Data type for output checkpoint file. Default: float16")
    args = parser.parse_args()

    # convert hf ckpt to ms
    convert_pt_to_ms(input_path=args.torch_path, output_path=args.mindspore_path, dtype=args.dtype)
