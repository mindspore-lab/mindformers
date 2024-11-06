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
transform huggingface model to mindspore ckpt.
"""

import os
import argparse

import numpy as np
import mindspore as ms

from mindformers.tools.utils import str2bool

dtype_map = {
    'fp32': ms.float32,
    'bf16': ms.bfloat16,
    'fp16': ms.float16
}


def name_replace(weight_name: str):
    """replace weight name"""
    weight_name = weight_name.replace('embed_tokens.', 'tok_embeddings.')
    weight_name = weight_name.replace('lm_head.', 'output.')
    weight_name = weight_name.replace('.self_attn.q_proj.', '.attention.wq.')
    weight_name = weight_name.replace('.self_attn.k_proj.', '.attention.wk.')
    weight_name = weight_name.replace('.self_attn.v_proj.', '.attention.wv.')
    weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')
    weight_name = weight_name.replace('.mlp.gate_proj.', '.feed_forward.w1.')
    weight_name = weight_name.replace('.mlp.down_proj.', '.feed_forward.w2.')
    weight_name = weight_name.replace('.mlp.up_proj.', '.feed_forward.w3.')
    weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
    weight_name = weight_name.replace(
        '.post_attention_layernorm.', '.ffn_norm.')
    return weight_name


# pylint: disable=W0613
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert hf weight to ms."""
    print(
        f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        from transformers import Qwen2ForCausalLM
    except ImportError as e:
        raise ImportError(
            "Failed to load HuggingFace checkpoint. "
            "Please make sure the 'transformers' library is installed and available."
        ) from e

    try:
        model_hf = Qwen2ForCausalLM.from_pretrained(
            os.path.dirname(input_path))
    # pylint: disable=W0703
    except Exception as e:
        print(
            f"Do not find huggingface checkpoint in '{os.path.dirname(input_path)}', Error {e.message}.", flush=True)
        return False
    ckpt_list = []
    for name, value in model_hf.state_dict().items():
        name = name_replace(name)
        if name == 'model.norm.weight':
            name = 'model.norm_out.weight'
        if name == 'output.weight':
            name = 'lm_head.weight'
        if name == 'model.tok_embeddings.weight':
            name = 'model.tok_embeddings.embedding_weight'
        value = value.detach().numpy()
        print(name, value.shape)
        ckpt_list.append({'name': name, 'data': ms.Tensor(value, dtype=dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    return True


def convert_qkv_concat_weight(param_dict):
    """convert qkv concat weight"""
    assume_num_layers = 500
    for i in range(assume_num_layers):
        # qkv weight concat
        wq_weight_name = f"model.layers.{i}.attention.wq.weight"
        wk_weight_name = f"model.layers.{i}.attention.wk.weight"
        wv_weight_name = f"model.layers.{i}.attention.wv.weight"
        qkv_concat_weight_name = f"model.layers.{i}.attention.w_qkv.weight"
        if wq_weight_name not in param_dict:
            break
        wq_weight = param_dict[wq_weight_name].asnumpy()
        wk_weight = param_dict[wk_weight_name].asnumpy()
        wv_weight = param_dict[wv_weight_name].asnumpy()
        qkv_weight = np.concatenate((wq_weight, wk_weight, wv_weight), 0)
        param_dict[qkv_concat_weight_name] = ms.Parameter(qkv_weight, name=qkv_concat_weight_name)

        # gate hidden weight concat
        ffn_gate_weight_name = f"model.layers.{i}.feed_forward.w1.weight"
        ffn_hidden_weight_name = f"model.layers.{i}.feed_forward.w3.weight"
        gate_hidden_concat_weight_name = f"model.layers.{i}.feed_forward.w_gate_hidden.weight"

        ffn_gate_weight = param_dict[ffn_gate_weight_name].asnumpy()
        ffn_hidden_weight = param_dict[ffn_hidden_weight_name].asnumpy()
        gate_hidden_weight = np.concatenate((ffn_gate_weight, ffn_hidden_weight), 0)
        param_dict[gate_hidden_concat_weight_name] = ms.Parameter(gate_hidden_weight,
                                                                  name=gate_hidden_concat_weight_name)

        param_dict.pop(wq_weight_name)
        param_dict.pop(wk_weight_name)
        param_dict.pop(wv_weight_name)
        param_dict.pop(ffn_gate_weight_name)
        param_dict.pop(ffn_hidden_weight_name)
        print("transform: {}".format(qkv_concat_weight_name))
        print("transform: {}".format(gate_hidden_concat_weight_name))

    for i in range(assume_num_layers):
        # qkv bias concat
        wq_bias_name = f"model.layers.{i}.attention.wq.bias"
        wk_bias_name = f"model.layers.{i}.attention.wk.bias"
        wv_bias_name = f"model.layers.{i}.attention.wv.bias"
        qkv_concat_bias_name = f"model.layers.{i}.attention.w_qkv.bias"
        if wq_bias_name not in param_dict:
            break

        wq_bias_weight = param_dict[wq_bias_name].asnumpy()
        wk_bias_weight = param_dict[wk_bias_name].asnumpy()
        wv_bias_weight = param_dict[wv_bias_name].asnumpy()
        qkv_bias_weight = np.concatenate((wq_bias_weight, wk_bias_weight, wv_bias_weight), 0)
        param_dict[qkv_concat_bias_name] = ms.Parameter(qkv_bias_weight, name=qkv_concat_bias_name)

        param_dict.pop(wq_bias_name)
        param_dict.pop(wk_bias_name)
        param_dict.pop(wv_bias_name)
        print("transform: {}".format(qkv_concat_bias_name))
    return param_dict


def convert_to_qkv_concat(pre_ckpt_path, mindspore_ckpt_path):
    """convert previous ckpt to qkv concat ckpt"""
    if os.path.isdir(pre_ckpt_path):
        rank_dir_list = os.listdir(pre_ckpt_path)
        for rank_dir in rank_dir_list:
            rank_dir_name = str(rank_dir)
            rank_id = rank_dir_name.split("_")[1]
            checkpoint_path = os.path.join(pre_ckpt_path, rank_dir_name, "checkpoint_{}.ckpt".format(rank_id))
            print("checkpoint_path: {}".format(checkpoint_path))
            params = ms.load_checkpoint(checkpoint_path)
            params = convert_qkv_concat_weight(params)

            save_dir = os.path.join(mindspore_ckpt_path, rank_dir_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(mindspore_ckpt_path, rank_dir_name, "checkpoint_{}.ckpt".format(rank_id))
            ms.save_checkpoint(params, save_path)
    else:
        params = ms.load_checkpoint(pre_ckpt_path)
        params = convert_qkv_concat_weight(params)
        ms.save_checkpoint(params, mindspore_ckpt_path)


def convert_weight(para):
    """convert weight entrance"""
    if not hasattr(para, 'torch_ckpt_dir'):
        para.torch_ckpt_dir = para.input_path
    if not hasattr(para, 'pre_ckpt_path'):
        para.pre_ckpt_path = para.input_path
    if not hasattr(para, 'mindspore_ckpt_path'):
        para.mindspore_ckpt_path = para.output_path
    if not para.dtype:
        para.dtype = "bf16"
    if para.qkv_concat:
        convert_to_qkv_concat(para.pre_ckpt_path, para.mindspore_ckpt_path)
    else:
        dtype = dtype_map.get(para.dtype)
        convert_pt_to_ms(input_path=para.torch_ckpt_dir,
                         output_path=para.mindspore_ckpt_path, dtype=dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_dir', default='./')
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    parser.add_argument('--pre_ckpt_path', default=None)
    parser.add_argument('--qkv_concat', default=False, type=str2bool)
    parser.add_argument('--dtype', default='bf16')
    args = parser.parse_args()

    convert_weight(args)
