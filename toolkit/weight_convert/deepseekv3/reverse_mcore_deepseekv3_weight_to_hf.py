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
"""Transform MindSpore Transformers MCore checkpoint to huggingface checkpoint."""

import os
import json
import argparse

from collections import defaultdict
from glob import glob
from time import time
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
    'num_routed_experts': 256,
    'num_layers': 61,
    'num_nextn_predict_layers': 1,
    'first_k_dense_replace': 3,
    'hidden_size': 7168,
    'ffn_hidden_size': 18432,
    'moe_ffn_hidden_size': 2048,
    'dtype': torch.bfloat16,
}


def split_linear_fc1_weight(linear_fc1_weight, ffn_hidden_size):
    """Split linear_fc1 to gate and up."""
    # 1. Process gate and up weight from discrete arrangement to continuous arrangement.
    target_shape = linear_fc1_weight.shape[0]
    idx = np.arange(target_shape)
    idx = np.concatenate((idx[::2], idx[1::2]), axis=0)
    linear_fc1_weight = linear_fc1_weight[idx]

    # 2. Split gate and up, then return them.
    return np.split(linear_fc1_weight, [ffn_hidden_size], axis=0)


def plain_name_replace(weight_name: str):
    """Weight name replacing for pre/post-process module"""
    weight_name = weight_name.replace('embedding.word_embeddings.weight', 'model.embed_tokens.weight')
    weight_name = weight_name.replace('decoder.final_layernorm.weight', 'model.norm.weight')
    weight_name = weight_name.replace('output_layer.weight', 'lm_head.weight')
    return weight_name


def mla_name_replace(weight_name: str, ms_layer_id, hf_layer_id):
    """Weight name replacing for MLA module weights"""
    weight_name = weight_name.replace(f'decoder.layers.{ms_layer_id}.', f'model.layers.{hf_layer_id}.')
    weight_name = weight_name.replace(f'mtp.layers.{ms_layer_id}.transformer_layer.', f'model.layers.{hf_layer_id}.')

    weight_name = weight_name.replace('.self_attention.linear_q_up_proj.', '.self_attn.q_b_proj.')
    weight_name = weight_name.replace('.self_attention.linear_kv_up_proj.', '.self_attn.kv_b_proj.')
    weight_name = weight_name.replace('.self_attention.linear_q_down_proj.', '.self_attn.q_a_proj.')
    weight_name = weight_name.replace('.self_attention.linear_kv_down_proj.', '.self_attn.kv_a_proj_with_mqa.')
    weight_name = weight_name.replace('.self_attention.q_layernorm.', '.self_attn.q_a_layernorm.')
    weight_name = weight_name.replace('.self_attention.kv_layernorm.', '.self_attn.kv_a_layernorm.')

    weight_name = weight_name.replace('.self_attention.linear_proj.', '.self_attn.o_proj.')

    return weight_name


def mlp_name_replace(weight_name: str, ms_layer_id, hf_layer_id):
    """Weight name replacing for MLP module, including MoE"""
    weight_name = weight_name.replace(f'decoder.layers.{ms_layer_id}.', f'model.layers.{hf_layer_id}.')
    weight_name = weight_name.replace(f'mtp.layers.{ms_layer_id}.transformer_layer.', f'model.layers.{hf_layer_id}.')

    weight_name = weight_name.replace('.pre_mlp_layernorm.', '.post_attention_layernorm.')

    # Dense MLP
    weight_name = weight_name.replace('.mlp.gating.', '.mlp.gate_proj.')
    weight_name = weight_name.replace('.mlp.linear_fc2.', '.mlp.down_proj.')
    weight_name = weight_name.replace('.mlp.hidden.', '.mlp.up_proj.')

    # MoE MLP
    weight_name = weight_name.replace('.mlp.shared_experts.gating.', '.mlp.shared_experts.gate_proj.')
    weight_name = weight_name.replace('.mlp.shared_experts.hidden.', '.mlp.shared_experts.up_proj.')
    weight_name = weight_name.replace('.mlp.shared_experts.linear_fc2.', '.mlp.shared_experts.down_proj.')

    weight_name = weight_name.replace('.mlp.router.weight', '.mlp.gate.weight')
    weight_name = weight_name.replace('.mlp.router.expert_bias', '.mlp.gate.e_score_correction_bias')

    return weight_name


def mtp_name_replace(weight_name: str, current_layer_id: int, mtp_layer_id: int):
    """replace weight name for MultiPredictionToken module"""
    weight_name = weight_name.replace(
        f"mtp.layers.{mtp_layer_id}.enorm", f"model.layers.{current_layer_id}.enorm"
    )
    weight_name = weight_name.replace(
        f"mtp.layers.{mtp_layer_id}.hnorm", f"model.layers.{current_layer_id}.hnorm"
    )
    weight_name = weight_name.replace(
        f"mtp.layers.{mtp_layer_id}.eh_proj", f"model.layers.{current_layer_id}.eh_proj"
    )
    weight_name = weight_name.replace(
        f"mtp.layers.{mtp_layer_id}.final_layernorm", f"model.layers.{current_layer_id}.shared_head.norm"
    )

    return weight_name


def load_data_ms(file_name):
    return ms.load_checkpoint(file_name, format="safetensors")


def layers_model_file_map(file_path, config):
    """
    Get the weight-file map dict of all the weight files
        where the corresponding weight is located according to each layer.
    """
    num_layers = config["num_layers"]
    layer_st_map = defaultdict(set)
    weight_map_file = os.path.join(file_path, "param_name_map.json")

    # Try to get the 'param_name_map' of weight.
    if os.path.exists(weight_map_file):
        with open(weight_map_file) as f:
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
        # Add MTP layers.
        elif weight_key.startswith("mtp.layers."):
            mtp_layer_name = int(weight_key.split('mtp.layers.')[1].split('.')[0])
            layer_name = num_layers + mtp_layer_name
            layer_st_map[layer_name].add(os.path.join(file_path, value))
        # Other weights, such as output_layer, word_embeddings, final_layernorm, and so on.
        else:
            layer_st_map[weight_key].add(os.path.join(file_path, value))
    return layer_st_map


def read_matched_file(layer_st_map, layer_list, is_first, is_last):
    """Load weights into dict for specified layers"""
    st_file_list = []
    for layer in layer_list:
        st_file_list.extend(list(layer_st_map[layer]))
    if is_first:
        st_file_list.extend(list(layer_st_map["embedding.word_embeddings.weight"]))
    if is_last:
        st_file_list.extend(list(layer_st_map["decoder.final_layernorm.weight"]))
        st_file_list.extend(list(layer_st_map["output_layer.weight"]))
    st_file_list = list(set(st_file_list))

    weights = {}
    for st_file in st_file_list:
        current_weight = load_data_ms(st_file)
        weights.update(current_weight)

    return weights


def _mla_ms_to_pt(layer_id, ms_layer_weights, config, is_mtp_layers=False):
    """Processing weights in MLA module"""
    dtype = config['dtype']

    num_layers = config['num_layers']
    hf_origin_layer_id = (
        (num_layers + layer_id)
        if is_mtp_layers
        else layer_id
    )

    layer_prefix = (
        # When use MTP layers, pass in the 'mtp_layer_id',
        # and the MTP layer id is (cur_layer_id - config['num_layers'])
        f"mtp.layers.{layer_id}.transformer_layer"
        if is_mtp_layers
        else f"decoder.layers.{layer_id}"
    )

    # Generate MLA Keys
    input_layernorm_key = f"{layer_prefix}.input_layernorm.weight"
    linear_q_down_proj_key = f"{layer_prefix}.self_attention.linear_q_down_proj.weight"
    linear_kv_down_proj_key = f"{layer_prefix}.self_attention.linear_kv_down_proj.weight"
    q_layernorm_key = f"{layer_prefix}.self_attention.q_layernorm.weight"
    kv_layernorm_key = f"{layer_prefix}.self_attention.kv_layernorm.weight"
    linear_q_up_proj_key = f"{layer_prefix}.self_attention.linear_q_up_proj.weight"
    linear_kv_up_proj_key = f"{layer_prefix}.self_attention.linear_kv_up_proj.weight"
    linear_proj_key = f"{layer_prefix}.self_attention.linear_proj.weight"

    # Get other MLA weights
    input_layernorm = cpu_cast(ms_layer_weights.pop(input_layernorm_key), ms.float32).numpy()
    linear_q_down_proj = cpu_cast(ms_layer_weights.pop(linear_q_down_proj_key), ms.float32).numpy()
    linear_kv_down_proj = cpu_cast(ms_layer_weights.pop(linear_kv_down_proj_key), ms.float32).numpy()
    linear_q_up_proj = cpu_cast(ms_layer_weights.pop(linear_q_up_proj_key), ms.float32).numpy()
    linear_kv_up_proj = cpu_cast(ms_layer_weights.pop(linear_kv_up_proj_key), ms.float32).numpy()
    q_layernorm = cpu_cast(ms_layer_weights.pop(q_layernorm_key), ms.float32).numpy()
    kv_layernorm = cpu_cast(ms_layer_weights.pop(kv_layernorm_key), ms.float32).numpy()
    linear_proj = cpu_cast(ms_layer_weights.pop(linear_proj_key), ms.float32).numpy()

    # Mapping the weight keys then add them into HF weight dict
    mla_weight_dict = defaultdict()

    hf_input_layernorm_key = mla_name_replace(input_layernorm_key, layer_id, hf_origin_layer_id)
    mla_weight_dict[hf_input_layernorm_key] = torch.from_numpy(input_layernorm).to(dtype).clone()

    q_a_proj_key = mla_name_replace(linear_q_down_proj_key, layer_id, hf_origin_layer_id)
    mla_weight_dict[q_a_proj_key] = torch.from_numpy(linear_q_down_proj).to(dtype).clone()

    q_a_layernorm_key = mla_name_replace(q_layernorm_key, layer_id, hf_origin_layer_id)
    mla_weight_dict[q_a_layernorm_key] = torch.from_numpy(q_layernorm).to(dtype).clone()

    q_b_proj_key = mla_name_replace(linear_q_up_proj_key, layer_id, hf_origin_layer_id)
    mla_weight_dict[q_b_proj_key] = torch.from_numpy(linear_q_up_proj).to(dtype).clone()

    kv_a_proj_with_mqa_key = mla_name_replace(linear_kv_down_proj_key, layer_id, hf_origin_layer_id)
    mla_weight_dict[kv_a_proj_with_mqa_key] = torch.from_numpy(linear_kv_down_proj).to(dtype).clone()

    kv_a_layernorm_key = mla_name_replace(kv_layernorm_key, layer_id, hf_origin_layer_id)
    mla_weight_dict[kv_a_layernorm_key] = torch.from_numpy(kv_layernorm).to(dtype).clone()

    kv_b_proj_key = mla_name_replace(linear_kv_up_proj_key, layer_id, hf_origin_layer_id)
    mla_weight_dict[kv_b_proj_key] = torch.from_numpy(linear_kv_up_proj).to(dtype).clone()

    o_proj_key = mla_name_replace(linear_proj_key, layer_id, hf_origin_layer_id)
    mla_weight_dict[o_proj_key] = torch.from_numpy(linear_proj).to(dtype).clone()

    return mla_weight_dict


def _mlp_ms_to_pt(layer_id, ms_layer_weights, config, is_mtp_layers=False):
    """Processing weights in MLP/MoE module"""
    dtype = config['dtype']
    num_layers = config['num_layers']
    first_k_dense_replace = config['first_k_dense_replace']
    num_routed_experts = config['num_routed_experts']

    hidden_size = config['hidden_size']
    ffn_hidden_size = config['ffn_hidden_size']
    moe_ffn_hidden_size = config['moe_ffn_hidden_size']

    hf_origin_layer_id = (
        (num_layers + layer_id)
        if is_mtp_layers
        else layer_id
    )

    layer_prefix = (
        # When use MTP layers, pass in the 'mtp_layer_id',
        # and the MTP layer id is (cur_layer_id - config['num_layers'])
        f"mtp.layers.{layer_id}.transformer_layer"
        if is_mtp_layers
        else f"decoder.layers.{layer_id}"
    )

    mlp_weight_dict = defaultdict()
    pre_mlp_layernorm_key = f"{layer_prefix}.pre_mlp_layernorm.weight"
    pre_mlp_layernorm = cpu_cast(ms_layer_weights.pop(pre_mlp_layernorm_key), ms.float32).numpy()
    post_attention_layernorm_key = mlp_name_replace(pre_mlp_layernorm_key, layer_id, hf_origin_layer_id)
    mlp_weight_dict[post_attention_layernorm_key] = torch.from_numpy(pre_mlp_layernorm).to(dtype).clone()

    if hf_origin_layer_id < first_k_dense_replace:
        # Dense MLP
        mlp_linear_fc1_key = f"{layer_prefix}.mlp.linear_fc1.weight"
        mlp_linear_fc2_key = f"{layer_prefix}.mlp.linear_fc2.weight"
        mlp_gating_key = f"{layer_prefix}.mlp.gating.weight"
        mlp_up_key = f"{layer_prefix}.mlp.hidden.weight"

        # Get ms weight
        mlp_linear_fc1 = cpu_cast(ms_layer_weights.pop(mlp_linear_fc1_key), ms.float32).numpy()
        mlp_linear_fc2 = cpu_cast(ms_layer_weights.pop(mlp_linear_fc2_key), ms.float32).numpy()

        # Process fc1 weight
        mlp_linear_gate, mlp_linear_up = split_linear_fc1_weight(
            linear_fc1_weight=mlp_linear_fc1,
            ffn_hidden_size=ffn_hidden_size,
        )

        # Replace keys
        gate_proj_key = mlp_name_replace(mlp_gating_key, layer_id, hf_origin_layer_id)
        up_proj_key = mlp_name_replace(mlp_up_key, layer_id, hf_origin_layer_id)
        down_proj_key = mlp_name_replace(mlp_linear_fc2_key, layer_id, hf_origin_layer_id)

        # Get HF weight
        mlp_weight_dict[gate_proj_key] = torch.from_numpy(mlp_linear_gate).to(dtype).clone()
        mlp_weight_dict[up_proj_key] = torch.from_numpy(mlp_linear_up).to(dtype).clone()
        mlp_weight_dict[down_proj_key] = torch.from_numpy(mlp_linear_fc2).to(dtype).clone()
    else:
        # MoE MLP
        mlp_router_weight_key = f"{layer_prefix}.mlp.router.weight"
        mlp_router_bias_key = f"{layer_prefix}.mlp.router.expert_bias"
        mlp_experts_weight1_key = f"{layer_prefix}.mlp.experts.weight1"
        mlp_experts_weight2_key = f"{layer_prefix}.mlp.experts.weight2"

        mlp_shared_experts_linear_fc1_key = f"{layer_prefix}.mlp.shared_experts.linear_fc1.weight"
        mlp_shared_experts_linear_fc2_key = f"{layer_prefix}.mlp.shared_experts.linear_fc2.weight"
        mlp_shared_experts_gating_key = f"{layer_prefix}.mlp.shared_experts.gating.weight"
        mlp_shared_experts_up_key = f"{layer_prefix}.mlp.shared_experts.hidden.weight"

        # Get ms weight
        mlp_router_weight = cpu_cast(ms_layer_weights.pop(mlp_router_weight_key), ms.float32).numpy()
        mlp_router_bias = cpu_cast(ms_layer_weights.pop(mlp_router_bias_key), ms.float32).numpy()
        mlp_experts_weight1 = cpu_cast(ms_layer_weights.pop(mlp_experts_weight1_key), ms.float32).numpy()
        mlp_experts_weight2 = cpu_cast(ms_layer_weights.pop(mlp_experts_weight2_key), ms.float32).numpy()

        mlp_shared_experts_linear_fc1 = cpu_cast(ms_layer_weights.pop(mlp_shared_experts_linear_fc1_key),
                                                 ms.float32).numpy()
        mlp_shared_experts_linear_fc2 = cpu_cast(ms_layer_weights.pop(mlp_shared_experts_linear_fc2_key),
                                                 ms.float32).numpy()

        # Process fc1 weight
        mlp_shared_experts_gate, mlp_shared_experts_up = split_linear_fc1_weight(
            linear_fc1_weight=mlp_shared_experts_linear_fc1,
            ffn_hidden_size=moe_ffn_hidden_size,
        )

        # Process experts weight1
        mlp_experts_weight1 = mlp_experts_weight1.reshape(num_routed_experts, hidden_size, moe_ffn_hidden_size * 2)
        # The shape of weight1 is (num_routed_experts, 2 * moe_ffn_hidden_size, hidden_size)
        mlp_experts_weight1 = mlp_experts_weight1.transpose(0, 2, 1)

        # Process experts weight2
        mlp_experts_weight2 = mlp_experts_weight2.reshape(num_routed_experts, moe_ffn_hidden_size, hidden_size)
        # The shape of weight2 is (num_routed_experts, hidden_size, moe_ffn_hidden_size)
        mlp_experts_weight2 = mlp_experts_weight2.transpose(0, 2, 1)

        # Split each expert weight.
        for i in range(num_routed_experts):
            # Generate current expert keys
            cur_expert_gate_key = f"model.layers.{hf_origin_layer_id}.mlp.experts.{i}.gate_proj.weight"
            cur_expert_up_key = f"model.layers.{hf_origin_layer_id}.mlp.experts.{i}.up_proj.weight"
            cur_expert_down_key = f"model.layers.{hf_origin_layer_id}.mlp.experts.{i}.down_proj.weight"

            cur_expert_weight1 = mlp_experts_weight1[i]
            # The shape of cur_expert_(gate/up) is (moe_ffn_hidden_size, hidden_size)
            cur_expert_gate, cur_expert_up = np.split(cur_expert_weight1, [moe_ffn_hidden_size], axis=0)
            # The shape of cur_expert_down is (hidden_size, moe_ffn_hidden_size)
            cur_expert_down = mlp_experts_weight2[i]

            mlp_weight_dict[cur_expert_gate_key] = torch.from_numpy(cur_expert_gate).to(dtype).clone().contiguous()
            mlp_weight_dict[cur_expert_up_key] = torch.from_numpy(cur_expert_up).to(dtype).clone().contiguous()
            mlp_weight_dict[cur_expert_down_key] = torch.from_numpy(cur_expert_down).to(dtype).clone().contiguous()

        # Replace keys
        gate_weight_key = mlp_name_replace(mlp_router_weight_key, layer_id, hf_origin_layer_id)
        gate_e_score_correction_bias_key = mlp_name_replace(mlp_router_bias_key, layer_id, hf_origin_layer_id)
        shared_experts_gate_key = mlp_name_replace(mlp_shared_experts_gating_key, layer_id, hf_origin_layer_id)
        shared_experts_up_key = mlp_name_replace(mlp_shared_experts_up_key, layer_id, hf_origin_layer_id)
        shared_experts_down_key = mlp_name_replace(mlp_shared_experts_linear_fc2_key, layer_id, hf_origin_layer_id)

        # Get the rest HF weight
        mlp_weight_dict[gate_weight_key] = torch.from_numpy(mlp_router_weight).to(dtype).clone()
        mlp_weight_dict[gate_e_score_correction_bias_key] = torch.from_numpy(mlp_router_bias).to(torch.float32).clone()
        mlp_weight_dict[shared_experts_gate_key] = torch.from_numpy(mlp_shared_experts_gate).to(dtype).clone()
        mlp_weight_dict[shared_experts_up_key] = torch.from_numpy(mlp_shared_experts_up).to(dtype).clone()
        mlp_weight_dict[shared_experts_down_key] = torch.from_numpy(mlp_shared_experts_linear_fc2).to(dtype).clone()

    return mlp_weight_dict


def _mtp_ms_to_pt(layer_id, ms_layer_weights, config):
    """Processing weights in MTP module"""
    num_layers = config["num_layers"]
    dtype = config['dtype']

    mtp_layer_id = layer_id - num_layers
    # ignore the shared emb_weights and lm head in mtp layers
    enorm_key = f"mtp.layers.{mtp_layer_id}.enorm.weight"
    hnorm_key = f"mtp.layers.{mtp_layer_id}.hnorm.weight"
    eh_proj_key = f"mtp.layers.{mtp_layer_id}.eh_proj.weight"
    final_layernorm_key = f"mtp.layers.{mtp_layer_id}.final_layernorm.weight"

    enorm = cpu_cast(ms_layer_weights.pop(enorm_key), ms.float32).numpy()
    hnorm = cpu_cast(ms_layer_weights.pop(hnorm_key), ms.float32).numpy()
    eh_proj = cpu_cast(ms_layer_weights.pop(eh_proj_key), ms.float32).numpy()
    shard_head_norm = cpu_cast(ms_layer_weights.pop(final_layernorm_key), ms.float32).numpy()

    mtp_weight_dict = defaultdict()
    enorm_key = mtp_name_replace(enorm_key, layer_id, mtp_layer_id)
    hnorm_key = mtp_name_replace(hnorm_key, layer_id, mtp_layer_id)
    eh_proj_key = mtp_name_replace(eh_proj_key, layer_id, mtp_layer_id)
    norm_out_key = mtp_name_replace(final_layernorm_key, layer_id, mtp_layer_id)

    # MTP norm weights
    mtp_weight_dict[enorm_key] = torch.from_numpy(enorm).to(dtype).clone()
    mtp_weight_dict[hnorm_key] = torch.from_numpy(hnorm).to(dtype).clone()
    mtp_weight_dict[eh_proj_key] = torch.from_numpy(eh_proj).to(dtype).clone()
    mtp_weight_dict[norm_out_key] = torch.from_numpy(shard_head_norm).to(dtype).clone()

    # MTP shared weights
    emb_weight_key = "embedding.word_embeddings.weight"
    lm_head_key = "output_layer.weight"
    emb_weight = cpu_cast(ms_layer_weights.get(emb_weight_key), ms.float32).numpy()
    lm_head = cpu_cast(ms_layer_weights.get(lm_head_key), ms.float32).numpy()

    shared_embed_key = f"model.layers.{layer_id}.embed_tokens.weight"
    shared_head_key = f"model.layers.{layer_id}.shared_head.head.weight"
    mtp_weight_dict[shared_embed_key] = torch.from_numpy(emb_weight).to(dtype).clone()
    mtp_weight_dict[shared_head_key] = torch.from_numpy(lm_head).to(dtype).clone()

    # MLA in MTP
    mtp_weight_dict.update(
        _mla_ms_to_pt(mtp_layer_id, ms_layer_weights, config, is_mtp_layers=True)
    )

    # MLP in MTP
    mtp_weight_dict.update(
        _mlp_ms_to_pt(mtp_layer_id, ms_layer_weights, config, is_mtp_layers=True)
    )

    return mtp_weight_dict


def _model_preprocess_ms_to_pt(ms_layer_weights, config):
    """Processing weights in prepross module"""
    dtype = config['dtype']
    emb_weight_key = "embedding.word_embeddings.weight"
    emb_weight = cpu_cast(ms_layer_weights.get(emb_weight_key), ms.float32).numpy()
    emb_weight_key = plain_name_replace(emb_weight_key)

    plain_weight_dict = defaultdict()
    plain_weight_dict[emb_weight_key] = torch.from_numpy(emb_weight).to(dtype).clone()

    return plain_weight_dict


def _model_postprocess_ms_to_pt(ms_layer_weights, config):
    """Processing weights in postpross module"""
    dtype = config['dtype']
    final_norm_key = "decoder.final_layernorm.weight"
    lm_head_key = "output_layer.weight"
    final_norm = cpu_cast(ms_layer_weights.get(final_norm_key), ms.float32).numpy()
    lm_head = cpu_cast(ms_layer_weights.get(lm_head_key), ms.float32).numpy()

    final_norm_key = plain_name_replace(final_norm_key)
    lm_head_key = plain_name_replace(lm_head_key)

    plain_weight_dict = defaultdict()
    plain_weight_dict[final_norm_key] = torch.from_numpy(final_norm).to(dtype).clone()
    plain_weight_dict[lm_head_key] = torch.from_numpy(lm_head).to(dtype).clone()

    return plain_weight_dict


def get_torch_storage_size(tensor):
    """Get tensor's storage size, requires torch >= 2.1"""
    return tensor.untyped_storage().nbytes()


def ms_safetensors_convertor(input_path, output_path, config):
    """Convert safetensors format checkpoint"""
    # Try to get weight-file map of each layer.
    layer_st_map = layers_model_file_map(input_path, config)

    num_layers = config["num_layers"]
    num_nextn_predict_layers = config["num_nextn_predict_layers"]
    total_num_layers = num_layers + num_nextn_predict_layers

    converted_st_map = defaultdict()
    converted_st_map["weight_map"] = defaultdict()
    converted_st_map["metadata"] = defaultdict()

    total_size = 0
    for layer_id in tqdm(
            range(total_num_layers), desc="Converting layers", unit="layers", position=0, leave=True
    ):
        # Get current layer weight keys.
        if layer_id == 0:
            ms_layer_weights = read_matched_file(layer_st_map, [layer_id], is_first=True, is_last=False)
        elif 0 < layer_id < num_layers - 1:
            ms_layer_weights = read_matched_file(layer_st_map, [layer_id], is_first=False, is_last=False)
        elif layer_id == num_layers - 1:
            ms_layer_weights = read_matched_file(layer_st_map, [layer_id], is_first=False, is_last=True)
        else:
            # For mtp layers, embed weight and lm_head weight are needed for shared weights.
            ms_layer_weights = read_matched_file(layer_st_map, [layer_id], is_first=True, is_last=True)

        pt_layer_weights = defaultdict()
        # First Layer
        if layer_id == 0:
            pt_layer_weights.update(
                _model_preprocess_ms_to_pt(ms_layer_weights, config)
            )
        # Last Layer
        if layer_id == total_num_layers - 1:
            pt_layer_weights.update(
                _model_postprocess_ms_to_pt(ms_layer_weights, config)
            )
        # MTP Layers process
        if layer_id > num_layers - 1:
            pt_layer_weights.update(
                _mtp_ms_to_pt(layer_id, ms_layer_weights, config)
            )
        else:
            pt_layer_weights.update(
                _mla_ms_to_pt(layer_id, ms_layer_weights, config)
            )
            pt_layer_weights.update(
                _mlp_ms_to_pt(layer_id, ms_layer_weights, config)
            )

        saving_file_name = f"model-{layer_id + 1:05d}-of-{total_num_layers:05d}.safetensors"
        for name in list(pt_layer_weights.keys()):
            converted_st_map["weight_map"][name] = saving_file_name
            total_size += get_torch_storage_size(pt_layer_weights.get(name))
        save_file(pt_layer_weights, os.path.join(output_path, saving_file_name))
        tqdm.write(f"Saved weights in layer-{layer_id} to file '{saving_file_name}' successfully!")

    converted_st_map["metadata"]["total_size"] = total_size
    converted_model_index_file = os.path.join(output_path, f"model.safetensors.index.json")
    with open(converted_model_index_file, "w") as f:
        json_string = json.dumps(converted_st_map, default=lambda x: x.__dict__, sort_keys=False, indent=2)
        f.write(json_string)
    set_safe_mode_for_file_or_dir(converted_model_index_file)
    tqdm.write(f"Param name map is saved into file '{converted_model_index_file}' successfully!")


def convert_ms_to_pt(input_path, output_path, config=None):
    """convert ms weight to huggingface."""
    if config is None:
        config = default_config
    os.makedirs(output_path, exist_ok=True)

    tqdm.write(f"Trying to convert huggingface checkpoint in '{input_path}'.")
    start_time = time()
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
        if key in ['num_layers', 'num_nextn_predict_layers', 'first_k_dense_replace',
                   'num_routed_experts', 'hidden_size', 'ffn_hidden_size', 'moe_ffn_hidden_size']:
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
    # Get configuration args
    parser = argparse.ArgumentParser()
    parser.add_argument('--huggingface_ckpt_path', default=None, type=str,
                        help="Converted HuggingFace checkpoint directory.")
    parser.add_argument('--mindspore_ckpt_path', default=None, type=str,
                        help="MindSpore Transformers MCore checkpoint directory.")

    parser.add_argument('--dtype', default='bf16', type=str, choices=['fp16', 'bf16', 'fp32'],
                        help="The dtype of converted weight, choices in ['fp16', 'bf16', 'fp32']")

    parser.add_argument("--num_layers", default=61, type=int,
                        help="The number of attention layers.")
    parser.add_argument("--num_nextn_predict_layers", default=1, type=int,
                        help="The number of Multi-Token Prediction layers.")
    parser.add_argument("--first_k_dense_replace", default=3, type=int,
                        help="Customizing the number of dense layers.")

    parser.add_argument('--num_routed_experts', default=256, type=int,
                        help="The number of routed experts.")
    parser.add_argument("--hidden_size", default=7168, type=int,
                        help="The size of Hidden layer.")
    parser.add_argument("--ffn_hidden_size", default=18432, type=int,
                        help="Transformer Feed-Forward Network hidden size.")
    parser.add_argument("--moe_ffn_hidden_size", default=2048, type=int,
                        help="MoE Feed-Forward Network hidden size.")

    args = parser.parse_args()

    reverse_weight(args)
