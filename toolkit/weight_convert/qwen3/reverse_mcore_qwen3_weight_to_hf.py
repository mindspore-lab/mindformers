# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Transform MindSpore Transformers MCore checkpoint of Qwen3 to huggingface checkpoint."""

import os
import json
import argparse

from collections import defaultdict
from glob import glob
from time import time
from functools import partial
from multiprocessing import Pool

from safetensors.torch import save_file
import numpy as np

from tqdm import tqdm
import torch

import mindspore as ms
from mindspore.ops.operations import Cast
from mindformers.tools.utils import set_safe_mode_for_file_or_dir
from mindformers.tools.logger import logger

ms.set_context(device_target='CPU')
cpu_cast = Cast().set_device('CPU')

DTYPE_MAP = {
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
    'fp16': torch.float16
}

DEFAULT_CONFIG = {
    'num_layers': 28,
    'kv_channels': 128,
    'num_query_groups': 8,
    'num_attention_heads': 16,
    'ffn_hidden_size': 3072,
    'dtype': torch.bfloat16,
    'max_worker': 16,
}


def split_qkv_weight(qkv_weight, head_dim, n_kv_heads, num_attention_heads):
    """
    Split qkv_weight to q_weight, k_weight, v_weight

    Args：
        qkv_weight: QKV weight,
            the shape is [((num_attention_heads // n_kv_heads) + 2) * head_dim * n_kv_heads, hidden_size].
        head_dim: Projection weights dimension in multi-head attention.
        n_kv_heads: Number of query groups for group query attention.
        num_attention_heads: Number of transformer attention heads.

    Returns：
        q_weight: shape = [n_kv_heads * n_rep * head_dim, hidden_size]
        k_weight: shape = [n_kv_heads * head_dim, hidden_size]
        v_weight: shape = [n_kv_heads * head_dim, hidden_size]
    """
    n_rep = num_attention_heads // n_kv_heads
    total_dim = (n_rep + 2) * head_dim

    concat_qkv_weight = qkv_weight.reshape(n_kv_heads, total_dim, -1)

    q_dim = n_rep * head_dim
    k_dim = head_dim

    q_part = concat_qkv_weight[:, :q_dim, :]
    k_part = concat_qkv_weight[:, q_dim:q_dim + k_dim, :]
    v_part = concat_qkv_weight[:, q_dim + k_dim:, :]

    q_weight = q_part.reshape(n_kv_heads * n_rep * head_dim, -1)
    k_weight = k_part.reshape(n_kv_heads * head_dim, -1)
    v_weight = v_part.reshape(n_kv_heads * head_dim, -1)

    return q_weight, k_weight, v_weight


def split_linear_fc1_weight(linear_fc1_weight, ffn_hidden_size):
    """Split linear_fc1 to gate and up."""
    # 1. Process gate and up weight from discrete arrangement to continuous arrangement.
    target_shape = linear_fc1_weight.shape[0]
    idx = np.arange(target_shape)
    idx = np.concatenate((idx[::2], idx[1::2]), axis=0)
    linear_fc1_weight = linear_fc1_weight[idx]

    # Verify the 'ffn_hidden_size' configuration,
    # and intercept if it does not meet expectations
    # to avoid precision issues caused by 'np.split' splitting into empty segments.
    if ffn_hidden_size != linear_fc1_weight.shape[0] / 2:
        raise ValueError(
            f"Split 'linear_fc1' weight failed! "
            f"The shape of 'linear_fc1' is {linear_fc1_weight.shape} now! "
            f"The 'ffn_hidden_size' is expected as 'linear_fc1.shape[0] / 2', but get {ffn_hidden_size}.\n"
            f"Please check your '--ffn_hidden_size' is configured correctly, "
            f"keep the same configuration as your yaml file. "
        )
    # 2. Split gate and up, then return them.
    return np.split(linear_fc1_weight, [ffn_hidden_size], axis=0)


def plain_name_replace(weight_name: str):
    """Weight name replacing for pre/post-process module"""
    weight_name = weight_name.replace('output_layer.weight', 'lm_head.weight')
    weight_name = weight_name.replace('embedding.word_embeddings.weight', 'model.embed_tokens.weight')
    weight_name = weight_name.replace('decoder.final_layernorm.weight', 'model.norm.weight')
    return weight_name


def mla_name_replace(weight_name: str):
    """Weight name replacing for MLA module weights"""
    weight_name = weight_name.replace('decoder.layers.', 'model.layers.')

    weight_name = weight_name.replace('.self_attention.q_layernorm.', '.self_attn.q_norm.')
    weight_name = weight_name.replace('.self_attention.k_layernorm.', '.self_attn.k_norm.')

    weight_name = weight_name.replace('.self_attention.linear_proj.', '.self_attn.o_proj.')

    return weight_name


def mlp_name_replace(weight_name: str):
    """Weight name replacing for MLP module, including MoE"""
    weight_name = weight_name.replace('decoder.layers.', 'model.layers.')

    weight_name = weight_name.replace('.pre_mlp_layernorm.', '.post_attention_layernorm.')

    weight_name = weight_name.replace('.mlp.gating.', '.mlp.gate_proj.')
    weight_name = weight_name.replace('.mlp.hidden.', '.mlp.up_proj.')
    weight_name = weight_name.replace('.mlp.linear_fc2.', '.mlp.down_proj.')

    return weight_name


def load_data_ms(file_name):
    return ms.load_checkpoint(file_name, format="safetensors")


def layers_model_file_map(file_path):
    """
    Get the weight-file map dict of all the weight files
        where the corresponding weight is located according to each layer.
    """
    layer_st_map = defaultdict(set)
    weight_map_file = os.path.join(file_path, "param_name_map.json")

    # Try to get the 'param_name_map' of weight.
    if os.path.exists(weight_map_file):
        with open(weight_map_file, encoding='utf-8') as f:
            weights_map = json.load(f)
        try:
            weights_map = weights_map["weight_map"]
        except KeyError:
            pass
    else:
        # Consider the scenario of only exits a single safetensors file without 'param_name_map'.
        logger.warning(f"Cannot find weight map file 'param_name_map.json' in path '{file_path}', "
                       f"Trying to load the single safetensors file ...")
        files = sorted(glob(os.path.join(file_path, "*.safetensors")))
        if not files:
            raise ValueError(f"No safetensors files found in path '{file_path}'.")

        # Get the file name of the first safetensors in the file list.
        weight_file = files[0].split("/")[-1]
        keys = load_data_ms(os.path.join(file_path, weight_file)).keys()
        # Get all keys of this single safetensors file as its weight mapping dict.
        weights_map = {}
        for k in keys:
            weights_map[k] = weight_file

    # Collect the file name corresponding to each layer.
    for weight_key, value in weights_map.items():
        # Add decoder layers, containing dense and MoE.
        if weight_key.startswith("decoder.layers."):
            layer_name = int(weight_key.split('decoder.layers.')[1].split('.')[0])
            layer_st_map[layer_name].add(os.path.join(file_path, value))
        # Other weights, such as output_layer, word_embeddings, final_layernorm, and so on.
        else:
            layer_st_map[weight_key].add(os.path.join(file_path, value))
    return layer_st_map


def read_matched_file(layer_st_map, layer_list, is_last):
    """Load weights into dict for specified layers"""
    st_file_list = []
    for layer in layer_list:
        st_file_list.extend(list(layer_st_map[layer]))
    if is_last:
        st_file_list.extend(list(layer_st_map["embedding.word_embeddings.weight"]))
        st_file_list.extend(list(layer_st_map["decoder.final_layernorm.weight"]))
        st_file_list.extend(list(layer_st_map["output_layer.weight"]))
    st_file_list = list(set(st_file_list))

    weights = {}
    for st_file in st_file_list:
        current_weight = load_data_ms(st_file)
        weights.update(current_weight)

    return weights


def _mla_ms_to_pt(layer_id, ms_layer_weights, config):
    """Processing weights in MLA module"""
    dtype = config['dtype']

    head_dim = config['kv_channels']
    n_kv_heads = config['num_query_groups']
    num_attention_heads = config['num_attention_heads']

    # Generate MLA Keys
    input_layernorm_key = f"decoder.layers.{layer_id}.input_layernorm.weight"
    linear_qkv_key = f"decoder.layers.{layer_id}.self_attention.linear_qkv.weight"
    q_layernorm_key = f"decoder.layers.{layer_id}.self_attention.q_layernorm.weight"
    k_layernorm_key = f"decoder.layers.{layer_id}.self_attention.k_layernorm.weight"
    linear_proj_key = f"decoder.layers.{layer_id}.self_attention.linear_proj.weight"

    # Get other MLA weights
    input_layernorm = cpu_cast(ms_layer_weights.pop(input_layernorm_key), ms.float32).numpy()
    linear_qkv = cpu_cast(ms_layer_weights.pop(linear_qkv_key), ms.float32).numpy()
    q_layernorm = cpu_cast(ms_layer_weights.pop(q_layernorm_key), ms.float32).numpy()
    k_layernorm = cpu_cast(ms_layer_weights.pop(k_layernorm_key), ms.float32).numpy()
    linear_proj = cpu_cast(ms_layer_weights.pop(linear_proj_key), ms.float32).numpy()

    # Mapping the weight keys then add them into HF weight dict
    mla_weight_dict = defaultdict()

    # Split QKV weight.
    hf_q_proj, hf_k_proj, hf_v_proj = split_qkv_weight(
        qkv_weight=linear_qkv,
        head_dim=head_dim,
        n_kv_heads=n_kv_heads,
        num_attention_heads=num_attention_heads
    )
    hf_q_proj_key = f"model.layers.{layer_id}.self_attn.q_proj.weight"
    mla_weight_dict[hf_q_proj_key] = torch.from_numpy(hf_q_proj).to(dtype).clone()
    hf_k_proj_key = f"model.layers.{layer_id}.self_attn.k_proj.weight"
    mla_weight_dict[hf_k_proj_key] = torch.from_numpy(hf_k_proj).to(dtype).clone()
    hf_v_proj_key = f"model.layers.{layer_id}.self_attn.v_proj.weight"
    mla_weight_dict[hf_v_proj_key] = torch.from_numpy(hf_v_proj).to(dtype).clone()

    # Other weight name replace.
    hf_input_layernorm_key = mla_name_replace(input_layernorm_key)
    mla_weight_dict[hf_input_layernorm_key] = torch.from_numpy(input_layernorm).to(dtype).clone()

    hf_q_norm_key = mla_name_replace(q_layernorm_key)
    mla_weight_dict[hf_q_norm_key] = torch.from_numpy(q_layernorm).to(dtype).clone()

    hf_k_norm_key = mla_name_replace(k_layernorm_key)
    mla_weight_dict[hf_k_norm_key] = torch.from_numpy(k_layernorm).to(dtype).clone()

    hf_o_proj_key = mla_name_replace(linear_proj_key)
    mla_weight_dict[hf_o_proj_key] = torch.from_numpy(linear_proj).to(dtype).clone()

    return mla_weight_dict


def _mlp_ms_to_pt(layer_id, ms_layer_weights, config):
    """Processing weights in MLP/MoE module"""
    dtype = config['dtype']
    ffn_hidden_size = config['ffn_hidden_size']

    mlp_weight_dict = defaultdict()
    pre_mlp_layernorm_key = f"decoder.layers.{layer_id}.pre_mlp_layernorm.weight"
    pre_mlp_layernorm = cpu_cast(ms_layer_weights.pop(pre_mlp_layernorm_key), ms.float32).numpy()
    post_attention_layernorm_key = mlp_name_replace(pre_mlp_layernorm_key)
    mlp_weight_dict[post_attention_layernorm_key] = torch.from_numpy(pre_mlp_layernorm).to(dtype).clone()

    # Dense MLP
    mlp_linear_fc1_key = f"decoder.layers.{layer_id}.mlp.linear_fc1.weight"
    mlp_linear_fc2_key = f"decoder.layers.{layer_id}.mlp.linear_fc2.weight"

    # Get ms weight
    mlp_linear_fc1 = cpu_cast(ms_layer_weights.pop(mlp_linear_fc1_key), ms.float32).numpy()
    mlp_linear_fc2 = cpu_cast(ms_layer_weights.pop(mlp_linear_fc2_key), ms.float32).numpy()

    # Process fc1 weight
    mlp_linear_gate, mlp_linear_up = split_linear_fc1_weight(
        linear_fc1_weight=mlp_linear_fc1,
        ffn_hidden_size=ffn_hidden_size,
    )

    # Replace keys
    mlp_gating_key = f"decoder.layers.{layer_id}.mlp.gating.weight"
    mlp_up_key = f"decoder.layers.{layer_id}.mlp.hidden.weight"

    gate_proj_key = mlp_name_replace(mlp_gating_key)
    up_proj_key = mlp_name_replace(mlp_up_key)
    down_proj_key = mlp_name_replace(mlp_linear_fc2_key)

    # Get HF weight
    mlp_weight_dict[gate_proj_key] = torch.from_numpy(mlp_linear_gate).to(dtype).clone()
    mlp_weight_dict[up_proj_key] = torch.from_numpy(mlp_linear_up).to(dtype).clone()
    mlp_weight_dict[down_proj_key] = torch.from_numpy(mlp_linear_fc2).to(dtype).clone()

    return mlp_weight_dict


def _model_postprocess_ms_to_pt(ms_layer_weights, config):
    """Processing weights in postprocess module"""
    dtype = config['dtype']

    lm_head_key = "output_layer.weight"
    emb_weight_key = "embedding.word_embeddings.weight"
    final_norm_key = "decoder.final_layernorm.weight"

    emb_weight = cpu_cast(ms_layer_weights.get(emb_weight_key), ms.float32).numpy()
    # When used 'tie_word_embedding' in train task,
    # 'embedding.word_embeddings.weight' will be shared with 'output_layer.weight',
    # and 'output_layer.weight' will not be saved.
    lm_head = (
        cpu_cast(ms_layer_weights.get(lm_head_key), ms.float32).numpy()
        if ms_layer_weights.get(lm_head_key) is not None
        else emb_weight
    )
    final_norm = cpu_cast(ms_layer_weights.get(final_norm_key), ms.float32).numpy()

    emb_weight_key = plain_name_replace(emb_weight_key)
    final_norm_key = plain_name_replace(final_norm_key)
    lm_head_key = plain_name_replace(lm_head_key)

    plain_weight_dict = defaultdict()
    plain_weight_dict[emb_weight_key] = torch.from_numpy(emb_weight).to(dtype).clone()
    plain_weight_dict[final_norm_key] = torch.from_numpy(final_norm).to(dtype).clone()
    plain_weight_dict[lm_head_key] = torch.from_numpy(lm_head).to(dtype).clone()

    return plain_weight_dict


def get_torch_storage_size(tensor):
    """Get tensor's storage size, requires torch >= 2.1"""
    return tensor.untyped_storage().nbytes()


def _process_single_layer(layer_id, *, num_layers, layer_st_map, output_path, config):
    """
    Processing a single layer facilitates multiprocess processing.
    """
    # Read the current layer weights.
    is_last = layer_id == num_layers - 1
    ms_layer_weights = read_matched_file(layer_st_map, [layer_id], is_last=is_last)

    pt_layer_weights = {}

    # Convert MLA and MLP weights
    pt_layer_weights.update(_mla_ms_to_pt(layer_id, ms_layer_weights, config))
    pt_layer_weights.update(_mlp_ms_to_pt(layer_id, ms_layer_weights, config))

    # The final layer involves additional processing (such as embedding).
    if is_last:
        pt_layer_weights.update(_model_postprocess_ms_to_pt(ms_layer_weights, config))

    # Construct save file name
    saving_file_name = f"model-{layer_id + 1:05d}-of-{num_layers:05d}.safetensors"
    file_path = os.path.join(output_path, saving_file_name)

    # Calculate the total size and build the `weight_map`.
    total_layer_size = 0
    weight_map_entries = {}
    for name, tensor in pt_layer_weights.items():
        total_layer_size += get_torch_storage_size(tensor)
        weight_map_entries[name] = saving_file_name

    # Save the weight file for this layer.
    save_file(pt_layer_weights, file_path)
    set_safe_mode_for_file_or_dir(file_path)

    return layer_id, weight_map_entries, total_layer_size, saving_file_name


def ms_safetensors_convertor(input_path, output_path, config):
    """Convert safetensors format checkpoint"""
    # Obtain the mapping from the original weight file to the layer.
    layer_st_map = layers_model_file_map(input_path)
    max_worker = config["max_worker"]
    num_layers = config["num_layers"]

    # Construct a working function with preset parameters (excluding layer_id).
    worker = partial(
        _process_single_layer,
        num_layers=num_layers,
        layer_st_map=layer_st_map,
        output_path=output_path,
        config=config
    )

    # Initialize the mapping json container.
    converted_st_map = {
        "weight_map": {},
        "metadata": {}
    }
    total_size = 0

    # Start the process pool to perform the conversion.
    with Pool(processes=max_worker) as pool:
        for result in tqdm(
                pool.imap_unordered(worker, range(num_layers)),
                total=num_layers,
                desc="Converting layers",
                leave=True,
                unit="layer"
        ):
            layer_id, weight_map_entries, layer_size, filename = result
            converted_st_map["weight_map"].update(weight_map_entries)
            total_size += layer_size
            tqdm.write(f"Saved layer-{layer_id} to '{filename}'")

    # Write the index file.
    converted_st_map["metadata"]["total_size"] = total_size
    converted_model_index_file = os.path.join(output_path, "model.safetensors.index.json")
    with open(converted_model_index_file, "w", encoding='utf-8') as f:
        json_string = json.dumps(converted_st_map, default=lambda x: x.__dict__, sort_keys=False, indent=2)
        f.write(json_string)
    set_safe_mode_for_file_or_dir(converted_model_index_file)
    tqdm.write(f"Param name map is saved into file '{converted_model_index_file}' successfully!")


def convert_ms_to_pt(input_path, output_path, config=None):
    """convert ms weight to huggingface."""
    if config is None:
        config = DEFAULT_CONFIG
    os.makedirs(output_path, exist_ok=True)

    start_time = time()
    tqdm.write(f"Trying to convert huggingface checkpoint in '{input_path}'.")
    print(f"Loading mindspore checkpoint in '{input_path}' ...")

    ms_safetensors_convertor(input_path, output_path, config)

    end_time = time()
    print("Finish converting mindspore checkpoints into Huggingface checkpoints!")
    tqdm.write(f"Cost time: {end_time - start_time}s.")


def reverse_weight(para):
    """convert weight entrance"""
    if not hasattr(para, 'mindspore_ckpt_path'):
        para.mindspore_ckpt_path = para.input_path
    if not hasattr(para, 'huggingface_ckpt_path'):
        para.huggingface_ckpt_path = para.output_path

    for key in DEFAULT_CONFIG:
        DEFAULT_CONFIG[key] = getattr(para, key, DEFAULT_CONFIG[key])
        if key in ['num_layers', 'num_attention_heads', 'num_query_groups', 'kv_channels',
                   'ffn_hidden_size', 'max_worker']:
            DEFAULT_CONFIG[key] = int(DEFAULT_CONFIG[key])

    DEFAULT_CONFIG['dtype'] = (
        DTYPE_MAP.get(DEFAULT_CONFIG['dtype'], DEFAULT_CONFIG['dtype'])
        if DEFAULT_CONFIG['dtype'] is not None
        else torch.bfloat16
    )

    convert_ms_to_pt(
        input_path=para.mindspore_ckpt_path,
        output_path=para.huggingface_ckpt_path,
        config=DEFAULT_CONFIG
    )


if __name__ == "__main__":
    # Get configuration args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--huggingface_ckpt_path', default=None, type=str,
                        help="Converted HuggingFace checkpoint directory.")
    parser.add_argument('--mindspore_ckpt_path', default=None, type=str,
                        help="MindSpore Transformers MCore checkpoint directory.")

    parser.add_argument('--dtype', default='bf16', type=str, choices=['fp16', 'bf16', 'fp32'],
                        help="The dtype of converted weight used, choices in ['fp16', 'bf16', 'fp32']")

    parser.add_argument("--num_layers", default=28, type=int,
                        help="The number of attention layers.")

    # For MLA to split QKV.
    parser.add_argument("--num_attention_heads", default=16, type=int,
                        help="Number of transformer attention heads.")
    parser.add_argument("--num_query_groups", default=8, type=int,
                        help="Number of query groups for group query attention.")
    parser.add_argument("--kv_channels", default=128, type=int,
                        help="Projection weights dimension in multi-head attention.")

    # For MLP to split gate and up.
    parser.add_argument("--ffn_hidden_size", default=3072, type=int,
                        help="Transformer Feed-Forward Network hidden size.")

    parser.add_argument("--max_worker", default=16, type=int,
                        help="Maximum number of child processes to be allocated. "
                             "Please manage child processes appropriately "
                             "to avoid resource contention caused by too many child processes, "
                             "as this may lead to OutOfMemoryError (OOM).")

    args = parser.parse_args()

    reverse_weight(args)
