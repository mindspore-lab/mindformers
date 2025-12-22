# Copyright 2025 TeleAI Technologies Co., Ltd
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
import argparse
from glob import glob

import torch
from safetensors.torch import load_file

import mindspore as ms
from mindformers.tools.utils import str2bool
from mindformers.tools import logger
from mindformers.utils.convert_utils import pt2ms

dtype_map = {
    'float32': ms.float32,
    'bfloat16': ms.bfloat16,
    'float16': ms.float16
}


def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('model.embed_tokens.weight', 'model.tok_embeddings.embedding_weight')
    name = name.replace('model.embedding_hidden_mapping_in.weight', 'model.embedding_hidden_mapping_in.weight')
    name = name.replace('model.embedding_hidden_mapping_out.weight', 'model.embedding_hidden_mapping_out.weight')
    name = name.replace('.input_layernorm', '.attention_norm')
    name = name.replace('.self_attn.o_proj.', '.attention.wo.')
    name = name.replace('.self_attn.q_proj.', '.attention.wq.')
    name = name.replace('.self_attn.k_proj.', '.attention.wk.')
    name = name.replace('.self_attn.v_proj.', '.attention.wv.')
    name = name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    name = name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    name = name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
    name = name.replace('lm_head.', 'lm_head.')
    name = name.replace('model.norm.', 'model.norm_out.')
    return name


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert telechat hf weight to ms."""
    files = list(glob(os.path.join(input_path, "pytorch_model*.bin")))
    convert_safetensors = False
    if not files:
        files = list(glob(os.path.join(input_path, "model*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No bin or safetensors found in the model path: {input_path}.")
        convert_safetensors = True
    files.sort()
    pt_states_list = []
    for per_file in files:
        if convert_safetensors:
            pt_states = load_file(per_file)
        else:
            pt_states = torch.load(per_file, map_location='cpu')
        pt_states_list.append(pt_states)

    ckpt_list = []
    for pt_states in pt_states_list:
        for name, value in pt_states.items():
            name = name_replace(name)
            if name.startswith('transformer.h.'):
                name = name.replace('transformer.h.', 'model.layers.')
            logger.info(f'\rprocessing parameter: {name} {value.shape}')
            ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    logger.info(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.")
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
    parser.add_argument("--dtype", default='float32', choices=['float16', 'float32', 'bfloat16'],
                        help="Data type for output checkpoint file. Default: float16")
    parser.add_argument('--qkv_concat', default=False, type=str2bool)
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    parser.add_argument('--pre_ckpt_path', default=None)
    parser.add_argument('--model_name', default="telechat_7B", type=str)
    args = parser.parse_args()
    ms_dtype = dtype_map.get(args.dtype)

    convert_pt_to_ms(input_path=args.torch_path, output_path=args.mindspore_path, dtype=ms_dtype)
