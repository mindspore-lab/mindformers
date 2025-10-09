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
"""Transform HuggingFace checkpoint of DeepSeekV3 to MindSpore Transformers MCore checkpoint."""

import os
import json
import argparse

from glob import glob
from time import time
from collections import defaultdict

from safetensors.torch import load_file
import torch
from tqdm import tqdm

import mindspore as ms
from mindformers.tools.logger import logger
from mindformers.tools.utils import set_safe_mode_for_file_or_dir

DTYPE_MAP = {
    'fp32': ms.float32,
    'bf16': ms.bfloat16,
    'fp16': ms.float16
}

DEFAULT_CONFIG = {
    'num_routed_experts': 256,
    'num_layers': 61,
    'num_nextn_predict_layers': 1,
    'first_k_dense_replace': 3,
    'hidden_size': 7168,
    'ffn_hidden_size': 18432,
    'moe_ffn_hidden_size': 2048,
    'dtype': ms.bfloat16,
}

DEFAULT_DTYPE = torch.bfloat16


def concat_linear_fc1(gate_weight, up_weight, ffn_hidden_size):
    """Concat gate and up to linear_fc1."""
    gate_reshape = gate_weight.reshape(ffn_hidden_size, 1, -1)
    up_reshape = up_weight.reshape(ffn_hidden_size, 1, -1)

    # Discrete arrangement
    linear_fc1 = torch.concat((gate_reshape, up_reshape), dim=1)

    linear_fc1 = linear_fc1.reshape(ffn_hidden_size * 2, -1)
    return linear_fc1


def trans_rope_weight(weight, qk_pos_emb_head_dim):
    """Process rope router weight."""
    w1 = weight[..., -qk_pos_emb_head_dim::2, :]
    w2 = weight[..., -qk_pos_emb_head_dim + 1::2, :]
    weight[..., -qk_pos_emb_head_dim:, :] = torch.concat((w1, w2), dim=-2)
    return weight


def str2bool(b: str):
    """String convert to Bool."""
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception(f"Invalid Bool Value: {b}.")
    return output


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor, efficiently handling cases where
    `weight` is not a multiple of `block_size` by broadcasting `scale`.

    Args:
        weight (ms.Tensor): The quantized weight tensor of shape (M, N).
        scale (ms.Tensor): The scale tensor of shape (M // block_size, N // block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        ms.Tensor: The dequantized weight tensor of the same shape as `weight`, converted to the default dtype.

    Raises:
        AssertionError: If `scale` dimensions do not align with `weight` shape after scaling.
    """
    if len(weight.shape) != 2:
        raise ValueError(f"Weight must be 2-dimensional like (M, N), but get '{weight.shape}'.")

    # Get the original dimensions of weight
    m, n = weight.shape

    # Compute the effective block dimensions for scale
    scale_m, scale_n = scale.shape
    weight_m = (m + block_size - 1) // block_size
    weight_n = (n + block_size - 1) // block_size

    if scale_m != weight_m:
        raise ValueError(f"Mismatch in scale rows({scale_m}) and weight rows({weight_m}).")
    if scale_n != weight_n:
        raise ValueError(f"Mismatch in scale columns({scale_n}) and weight columns({weight_n}).")

    # Convert weight to float32 for calculations
    weight = weight.to(torch.float32)

    # Expand scale to match the weight tensor's shape
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:m, :n]

    # Perform element-wise multiplication
    dequantized_weight = weight * scale_expanded

    # Convert the output to the default dtype
    dequantized_weight = dequantized_weight.to(DEFAULT_DTYPE)

    return dequantized_weight


def dequant_layer_weights(layer_id, hf_layer_weights):
    """Dequanting weights in a layer"""
    dequanted_weights = {}
    for weight_name, weight in tqdm(hf_layer_weights.items(),
                                    desc=f"Dequant weights of layer-{layer_id}", unit="value", position=3, leave=True):
        if weight_name.endswith("_scale_inv"):
            continue
        elif weight.element_size() == 1 and (f"model.layers.{layer_id}." in weight_name):  # FP8 weight
            scale_inv_name = f"{weight_name}_scale_inv"
            try:
                # Get scale_inv from the correct file
                scale_inv = hf_layer_weights.get(scale_inv_name)
                dequantized_weight = weight_dequant(weight, scale_inv)
                dequanted_weights[weight_name] = dequantized_weight
            except KeyError:
                tqdm.write(f"Warning: Missing scale_inv tensor for {weight_name}, skipping dequanting")
                dequanted_weights[weight_name] = weight
        else:
            dequanted_weights[weight_name] = weight
    return dequanted_weights


def load_hf_data(file_name):
    """Load HuggingFace weight data"""
    return load_file(file_name)


def get_hf_layers_model_file_map(file_path):
    """Get weight-file map"""
    hf_param_name_map = defaultdict(set)
    weight_map_file = os.path.join(file_path, "model.safetensors.index.json")
    if os.path.exists(weight_map_file):
        with open(weight_map_file) as f:
            weights_map = json.load(f)
        weights_map = weights_map["weight_map"]
    else:
        logger.warning(f"Cannot find weight map file model.safetensors.index.json in path {file_path}, "
                       f"Trying to load one safetensor file ...")
        files = sorted(glob(os.path.join(file_path, "*.safetensors")))
        if not files:
            raise ValueError(f"No safetensors files found in path '{file_path}'.")

        weight_file = files[0].split("/")[-1]
        keys = load_hf_data(os.path.join(file_path, weight_file)).keys()
        weights_map = {}
        for k in keys:
            weights_map[k] = weight_file

    for weight_key, value in weights_map.items():
        if weight_key.startswith("model.layers."):
            layer_name = int(weight_key.split('model.layers.')[1].split('.')[0])
            hf_param_name_map[layer_name].add(os.path.join(file_path, value))
        else:
            hf_param_name_map[weight_key].add(os.path.join(file_path, value))
    return hf_param_name_map


def read_matched_hf_file(hf_param_name_map, layer_list, is_first, is_last):
    """Load weights into dict for specified layers"""
    st_file_list = []
    for layer in layer_list:
        st_file_list.extend(list(hf_param_name_map[layer]))
    if is_first:
        st_file_list.extend(list(hf_param_name_map["model.embed_tokens.weight"]))
    if is_last:
        st_file_list.extend(list(hf_param_name_map["model.norm.weight"]))
        st_file_list.extend(list(hf_param_name_map["lm_head.weight"]))
    st_file_list = list(set(st_file_list))
    weights = {}
    for st_file in tqdm(st_file_list,
                        desc="Reading weights from file", unit="value", position=2, leave=True):
        current_weight = load_hf_data(st_file)
        weights.update(current_weight)
    return weights


def _trans_model_layer_norm_hf_to_ms(hf_layer_weights, layer_id, mtp_layer_id=-1, is_mtp_layers: bool = False):
    """Layer norm process"""
    hf_input_norm_key = f"model.layers.{layer_id}.input_layernorm.weight"
    hf_post_attn_norm_key = f"model.layers.{layer_id}.post_attention_layernorm.weight"

    hf_input_norm_weight = hf_layer_weights.pop(hf_input_norm_key)
    hf_post_attn_norm_weight = hf_layer_weights.pop(hf_post_attn_norm_key)

    ms_input_norm_key = (
        f"mtp.layers.{mtp_layer_id}.transformer_layer.input_layernorm.weight"
        if is_mtp_layers
        else f"decoder.layers.{layer_id}.input_layernorm.weight"
    )
    ms_post_attn_norm_key = (
        f"mtp.layers.{mtp_layer_id}.transformer_layer.pre_mlp_layernorm.weight"
        if is_mtp_layers
        else f"decoder.layers.{layer_id}.pre_mlp_layernorm.weight"
    )

    model_layer_norm_dict = defaultdict()
    model_layer_norm_dict[ms_input_norm_key] = hf_input_norm_weight.clone()
    model_layer_norm_dict[ms_post_attn_norm_key] = hf_post_attn_norm_weight.clone()

    return model_layer_norm_dict


def _trans_model_layer_attn_hf_to_ms(hf_layer_weights, layer_id, mtp_layer_id=-1, is_mtp_layers: bool = False):
    """Attention layer process"""
    # Get HF attention layers keys
    hf_dense_key = f"model.layers.{layer_id}.self_attn.o_proj.weight"
    hf_q_layernorm_key = f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight"
    hf_kv_layernorm_key = f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"
    hf_q_a_proj_key = f"model.layers.{layer_id}.self_attn.q_a_proj.weight"
    hf_kv_a_proj_key = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"
    hf_q_b_proj_key = f"model.layers.{layer_id}.self_attn.q_b_proj.weight"
    hf_kv_b_proj_key = f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"

    # Get MS attention layers keys
    attn_layers_prefix = (
        f"mtp.layers.{mtp_layer_id}.transformer_layer"
        if is_mtp_layers
        else f"decoder.layers.{layer_id}"
    )
    ms_dense_key = f"{attn_layers_prefix}.self_attention.linear_proj.weight"
    ms_q_layernorm_key = f"{attn_layers_prefix}.self_attention.q_layernorm.weight"
    ms_kv_layernorm_key = f"{attn_layers_prefix}.self_attention.kv_layernorm.weight"
    ms_q_up_key = f"{attn_layers_prefix}.self_attention.linear_q_up_proj.weight"
    ms_kv_up_key = f"{attn_layers_prefix}.self_attention.linear_kv_up_proj.weight"
    ms_q_down_key = f"{attn_layers_prefix}.self_attention.linear_q_down_proj.weight"
    ms_kv_down_key = f"{attn_layers_prefix}.self_attention.linear_kv_down_proj.weight"

    # Get HF weight
    hf_dense_weight = hf_layer_weights.pop(hf_dense_key)
    hf_q_layernorm_weight = hf_layer_weights.pop(hf_q_layernorm_key)
    hf_kv_layernorm_weight = hf_layer_weights.pop(hf_kv_layernorm_key)
    hf_q_up_proj_weight = hf_layer_weights.pop(hf_q_b_proj_key)
    hf_kv_up_proj_weight = hf_layer_weights.pop(hf_kv_b_proj_key)
    hf_q_down_weight = hf_layer_weights.pop(hf_q_a_proj_key)
    hf_kv_down_weight = hf_layer_weights.pop(hf_kv_a_proj_key)

    # Set converted attention layers weight
    model_layer_attn_dict = defaultdict()
    model_layer_attn_dict[ms_dense_key] = hf_dense_weight.clone()
    model_layer_attn_dict[ms_q_layernorm_key] = hf_q_layernorm_weight.clone()
    model_layer_attn_dict[ms_kv_layernorm_key] = hf_kv_layernorm_weight.clone()
    model_layer_attn_dict[ms_q_up_key] = hf_q_up_proj_weight.clone()
    model_layer_attn_dict[ms_kv_up_key] = hf_kv_up_proj_weight.clone()
    model_layer_attn_dict[ms_q_down_key] = hf_q_down_weight.clone()
    model_layer_attn_dict[ms_kv_down_key] = hf_kv_down_weight.clone()

    return model_layer_attn_dict


def _trans_model_layer_mlp_hf_to_ms(hf_layer_weights, config, layer_id, mtp_layer_id=-1,
                                    is_mtp_layers: bool = False):
    """MLP layer process"""
    # Get config value
    first_k_dense_replace = config['first_k_dense_replace']
    num_routed_experts = config['num_routed_experts']

    hidden_size = config['hidden_size']
    ffn_hidden_size = config['ffn_hidden_size']
    moe_ffn_hidden_size = config['moe_ffn_hidden_size']

    model_layer_mlp_dict = defaultdict()

    if layer_id < first_k_dense_replace:
        # dense layer
        hf_experts_gate_proj_key = f"model.layers.{layer_id}.mlp.gate_proj.weight"
        hf_experts_up_proj_key = f"model.layers.{layer_id}.mlp.up_proj.weight"
        hf_linear_fc2_key = f"model.layers.{layer_id}.mlp.down_proj.weight"

        hf_experts_gate_proj_weight = hf_layer_weights.pop(hf_experts_gate_proj_key)
        hf_experts_up_proj_weight = hf_layer_weights.pop(hf_experts_up_proj_key)
        hf_linear_fc1_weight = concat_linear_fc1(hf_experts_gate_proj_weight, hf_experts_up_proj_weight,
                                                 ffn_hidden_size)
        hf_linear_fc2_weight = hf_layer_weights.pop(hf_linear_fc2_key)

        ms_linear_fc1_key = f"decoder.layers.{layer_id}.mlp.linear_fc1.weight"
        ms_linear_fc2_key = f"decoder.layers.{layer_id}.mlp.linear_fc2.weight"

        model_layer_mlp_dict[ms_linear_fc1_key] = hf_linear_fc1_weight.clone()
        model_layer_mlp_dict[ms_linear_fc2_key] = hf_linear_fc2_weight.clone()
    else:
        # MoE layer & MTP layer

        # Get MLP layer keys
        hf_mlp_router_weight_key = f"model.layers.{layer_id}.mlp.gate.weight"
        hf_mlp_router_bias_key = f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias"
        hf_shared_gate_proj_key = f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
        hf_shared_up_proj_key = f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"
        hf_shared_fc2_key = f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight"

        # Get MLP layer weights
        hf_mlp_router_weight = hf_layer_weights.pop(hf_mlp_router_weight_key)
        hf_mlp_router_bias_weight = hf_layer_weights.pop(hf_mlp_router_bias_key)

        hf_shared_gate_proj_weight = hf_layer_weights.pop(hf_shared_gate_proj_key)
        hf_shared_up_proj_weight = hf_layer_weights.pop(hf_shared_up_proj_key)

        hf_shared_fc1_weight = concat_linear_fc1(
            hf_shared_gate_proj_weight, hf_shared_up_proj_weight, moe_ffn_hidden_size
        )
        hf_shared_fc2_weight = hf_layer_weights.pop(hf_shared_fc2_key)

        experts_linear_fc1_list = []
        experts_linear_fc2_list = []

        # Process experts weight
        for expert_id in range(num_routed_experts):
            hf_experts_gate_proj_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
            hf_experts_up_proj_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
            hf_experts_fc2_key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"

            hf_experts_gate_proj_weight = hf_layer_weights.pop(hf_experts_gate_proj_key)
            hf_experts_up_proj_weight = hf_layer_weights.pop(hf_experts_up_proj_key)

            experts_fc1_weight = torch.concat(
                (hf_experts_gate_proj_weight, hf_experts_up_proj_weight),
                dim=0
            )
            experts_fc2_weight = hf_layer_weights.pop(hf_experts_fc2_key)

            experts_linear_fc1_list.append(experts_fc1_weight)
            experts_linear_fc2_list.append(experts_fc2_weight)

        # Generate experts weights key
        mlp_router_prefix = (
            f"mtp.layers.{mtp_layer_id}.transformer_layer"
            if is_mtp_layers
            else f"decoder.layers.{layer_id}"
        )
        ms_router_key = f"{mlp_router_prefix}.mlp.router.weight"
        ms_router_bias_key = f"{mlp_router_prefix}.mlp.router.expert_bias"
        ms_shared_fc1_key = f"{mlp_router_prefix}.mlp.shared_experts.linear_fc1.weight"
        ms_shared_fc2_key = f"{mlp_router_prefix}.mlp.shared_experts.linear_fc2.weight"

        # Set converted mlp layer weights
        model_layer_mlp_dict = defaultdict()
        model_layer_mlp_dict[ms_router_key] = hf_mlp_router_weight.clone()
        model_layer_mlp_dict[ms_router_bias_key] = hf_mlp_router_bias_weight.clone()
        model_layer_mlp_dict[ms_shared_fc1_key] = hf_shared_fc1_weight.clone()
        model_layer_mlp_dict[ms_shared_fc2_key] = hf_shared_fc2_weight.clone()

        # Get experts weight1/2 keys
        ms_experts_weight1_key = f"{mlp_router_prefix}.mlp.experts.weight1"
        ms_experts_weight2_key = f"{mlp_router_prefix}.mlp.experts.weight2"

        # Use GEMM, experts weight should be concatenated
        experts_linear_fc1 = torch.stack(experts_linear_fc1_list, dim=0)
        experts_linear_fc1 = torch.permute(experts_linear_fc1, (0, 2, 1))
        # The shape of experts_linear_fc1 is (num_routed_experts * hiddensize, 2 * moe_ffn_hidden_size)
        experts_linear_fc1 = experts_linear_fc1.reshape(num_routed_experts * hidden_size, -1)

        experts_linear_fc2 = torch.stack(experts_linear_fc2_list, dim=0)
        experts_linear_fc2 = torch.permute(experts_linear_fc2, (0, 2, 1))
        # The shape of experts_linear_fc2 is (num_routed_experts * moe_ffn_hidden_size, hidden_size)
        experts_linear_fc2 = experts_linear_fc2.reshape(-1, hidden_size)

        # Set GEMM experts weight
        model_layer_mlp_dict[ms_experts_weight1_key] = experts_linear_fc1.clone()
        model_layer_mlp_dict[ms_experts_weight2_key] = experts_linear_fc2.clone()

    return model_layer_mlp_dict


def _mtp_preprocess_hf_to_ms(hf_layer_weights, layer_id, mtp_layer_id):
    """Processing weights in prepross MTP layers"""
    hf_enorm_key = f"model.layers.{layer_id}.enorm.weight"
    hf_hnorm_key = f"model.layers.{layer_id}.hnorm.weight"
    hf_eh_proj_key = f"model.layers.{layer_id}.eh_proj.weight"
    hf_emb_key = f"model.layers.{layer_id}.embed_tokens.weight"

    hf_enorm_weight = hf_layer_weights.pop(hf_enorm_key)
    hf_hnorm_weight = hf_layer_weights.pop(hf_hnorm_key)
    hf_eh_proj_weight = hf_layer_weights.pop(hf_eh_proj_key)
    hf_emb_weight = hf_layer_weights.pop(hf_emb_key)

    ms_enorm_key = f"mtp.layers.{mtp_layer_id}.enorm.weight"
    ms_hnorm_key = f"mtp.layers.{mtp_layer_id}.hnorm.weight"
    ms_eh_proj_key = f"mtp.layers.{mtp_layer_id}.eh_proj.weight"
    ms_emb_key = f"embedding.word_embeddings.weight"

    preprocess_mtp_dict = defaultdict()
    preprocess_mtp_dict[ms_enorm_key] = hf_enorm_weight.clone()
    preprocess_mtp_dict[ms_hnorm_key] = hf_hnorm_weight.clone()
    preprocess_mtp_dict[ms_eh_proj_key] = hf_eh_proj_weight.clone()
    preprocess_mtp_dict[ms_emb_key] = hf_emb_weight.clone()

    return preprocess_mtp_dict


def _mtp_postprocess_hf_to_ms(hf_layer_weights, layer_id, mtp_layer_id):
    """Processing weights in postpross MTP layers"""
    hf_mtp_norm_key = f"model.layers.{layer_id}.shared_head.norm.weight"

    hf_mtp_norm_weight = hf_layer_weights.pop(hf_mtp_norm_key)

    ms_mtp_norm_key = f"mtp.layers.{mtp_layer_id}.final_layernorm.weight"

    preprocess_mtp_dict = defaultdict()
    preprocess_mtp_dict[ms_mtp_norm_key] = hf_mtp_norm_weight.clone()

    return preprocess_mtp_dict


def _mtp_hf_to_ms(layer_id, hf_layer_weights, config):
    """Processing weights in MTP module, the shared weights will not be ignored"""
    # Get MTP layers id
    num_layers = config["num_layers"]
    mtp_layer_id = layer_id - num_layers

    mtp_weight_dict = defaultdict()
    # preprocess MTP layers
    mtp_weight_dict.update(
        _mtp_preprocess_hf_to_ms(hf_layer_weights, layer_id, mtp_layer_id)
    )
    # process norm/attn/mlp of MTP layers
    mtp_weight_dict.update(
        _trans_model_layer_norm_hf_to_ms(hf_layer_weights, layer_id, mtp_layer_id, is_mtp_layers=True)
    )
    mtp_weight_dict.update(
        _trans_model_layer_attn_hf_to_ms(hf_layer_weights, layer_id, mtp_layer_id, is_mtp_layers=True)
    )
    mtp_weight_dict.update(
        _trans_model_layer_mlp_hf_to_ms(hf_layer_weights, config, layer_id, mtp_layer_id, is_mtp_layers=True)
    )
    # postprocess MTP layers
    mtp_weight_dict.update(
        _mtp_postprocess_hf_to_ms(hf_layer_weights, layer_id, mtp_layer_id)
    )

    return mtp_weight_dict


def _model_preprocess_hf_to_ms(hf_layer_weights):
    """Processing weights in prepross module"""
    hf_emb_weight_key = "model.embed_tokens.weight"
    hf_emb_weight = hf_layer_weights.pop(hf_emb_weight_key)

    ms_emb_weight_key = "embedding.word_embeddings.weight"

    preprocess_weight_dict = defaultdict()
    preprocess_weight_dict[ms_emb_weight_key] = hf_emb_weight.clone()

    return preprocess_weight_dict


def _model_postprocess_hf_to_ms(hf_layer_weights, has_mtp_layers=False):
    """Processing weights in postpross module"""
    hf_final_norm_key = "model.norm.weight"
    hf_lm_head_key = "lm_head.weight"
    hf_final_norm = hf_layer_weights.pop(hf_final_norm_key)
    hf_lm_head = hf_layer_weights.pop(hf_lm_head_key)

    ms_final_norm_key = "decoder.final_layernorm.weight"
    ms_lm_head_key = "output_layer.weight"

    postprocess_weight_dict = defaultdict()
    postprocess_weight_dict[ms_final_norm_key] = hf_final_norm.clone()
    postprocess_weight_dict[ms_lm_head_key] = hf_lm_head.clone()

    if has_mtp_layers:
        mtp_lm_head_key = "mtp.output_layer.weight"
        postprocess_weight_dict[mtp_lm_head_key] = hf_lm_head

    return postprocess_weight_dict


def ms_safetensors_convertor(input_path, output_path, config):
    """Convert to safetensors format checkpoint"""
    hf_param_name_map = get_hf_layers_model_file_map(input_path)

    dtype = config["dtype"]
    num_layers = config["num_layers"]
    num_nextn_predict_layers = config["num_nextn_predict_layers"]

    has_mtp_layers = num_nextn_predict_layers == 0
    total_num_layers = num_layers + num_nextn_predict_layers

    converted_param_name_map = defaultdict()
    for layer_id in tqdm(range(total_num_layers),
                         desc="Converting layers", unit="layers", position=1, leave=True):
        # Get a layer weight of huggingface weight
        if layer_id == 0:
            hf_layer_weights = read_matched_hf_file(
                hf_param_name_map=hf_param_name_map, layer_list=[layer_id], is_first=True, is_last=False
            )
        elif layer_id == total_num_layers - 1:
            hf_layer_weights = read_matched_hf_file(
                hf_param_name_map=hf_param_name_map, layer_list=[layer_id], is_first=False, is_last=True
            )
        else:
            hf_layer_weights = read_matched_hf_file(
                hf_param_name_map=hf_param_name_map, layer_list=[layer_id], is_first=False, is_last=False
            )

        # Dequant weights firstly
        hf_layer_weights = dequant_layer_weights(layer_id, hf_layer_weights)

        # Get the replaced weight name and corresponding value in ms
        ms_layer_weights = defaultdict()

        # pre/post-process of the DeepSeekV3 model weight
        if layer_id == 0:
            ms_layer_weights.update(
                _model_preprocess_hf_to_ms(hf_layer_weights)
            )
        if layer_id == total_num_layers - 1:
            ms_layer_weights.update(
                _model_postprocess_hf_to_ms(hf_layer_weights, has_mtp_layers)
            )
        # MTP Layers process
        if layer_id > num_layers - 1:
            ms_layer_weights.update(
                _mtp_hf_to_ms(layer_id, hf_layer_weights, config)
            )
        # no MTP layers process
        else:
            # MLA Layers process
            ms_layer_weights.update(
                _trans_model_layer_norm_hf_to_ms(hf_layer_weights, layer_id)
            )
            ms_layer_weights.update(
                _trans_model_layer_attn_hf_to_ms(hf_layer_weights, layer_id)
            )
            # MLP Layers process
            ms_layer_weights.update(
                _trans_model_layer_mlp_hf_to_ms(hf_layer_weights, config, layer_id)
            )

        # Process this layer weights' saving information
        to_save_ckpt = []
        saving_file = f"ms-model-{layer_id + 1:05d}-of-{total_num_layers:05d}.safetensors"
        for name in tqdm(list(ms_layer_weights.keys()),
                         desc=f"Saving weights in layer-{layer_id}.", unit="value", position=4, leave=True):
            value = ms_layer_weights.pop(name)
            value = value.to(torch.float32).numpy()

            tmp_dtype = dtype
            if "expert_bias" in name:
                tmp_dtype = ms.float32
            to_save_ckpt.append(
                {
                    'name': name,
                    'data': ms.Tensor(value, dtype=tmp_dtype)
                }
            )

            # Write the converted weight key and corresponding file to the param_name_map
            converted_param_name_map[name] = saving_file

        ms.save_checkpoint(to_save_ckpt, os.path.join(output_path, saving_file), format='safetensors')
        tqdm.write(f"Saved weights in layer-{layer_id} to file '{saving_file}' successfully!")

    # Writing the param_name_map.json
    converted_model_index_file = os.path.join(output_path, f"param_name_map.json")
    with open(converted_model_index_file, "w") as f:
        json_string = json.dumps(converted_param_name_map, default=lambda x: x.__dict__, sort_keys=False, indent=2)
        f.write(json_string)
    set_safe_mode_for_file_or_dir(converted_model_index_file)
    tqdm.write(f"Param name map is saved into file '{converted_model_index_file}' successfully!")


def convert_hf_to_ms(input_path, output_path, config=None):
    """convert HuggingFace weight to MindSpore Transformers."""
    if config is None:
        config = DEFAULT_CONFIG
    os.makedirs(output_path, exist_ok=True)

    tqdm.write(f"Trying to convert huggingface checkpoint in '{input_path}'.")
    start_time = time()

    ms_safetensors_convertor(input_path, output_path, config)

    end_time = time()
    tqdm.write("Finish converting Huggingface checkpoints into mindspore checkpoints!")
    tqdm.write(f"Cost time: {end_time - start_time}s.")


def convert_weight(para):
    """convert weight entrance"""
    if not hasattr(para, 'huggingface_ckpt_path'):
        para.huggingface_ckpt_path = para.input_path
    if not hasattr(para, 'mindspore_ckpt_path'):
        para.mindspore_ckpt_path = para.output_path

    for key in DEFAULT_CONFIG:
        DEFAULT_CONFIG[key] = getattr(para, key, DEFAULT_CONFIG[key])
        if key in ['num_routed_experts', 'num_layers', 'num_nextn_predict_layers', 'first_k_dense_replace',
                   'hidden_size', 'ffn_hidden_size', 'moe_ffn_hidden_size']:
            DEFAULT_CONFIG[key] = int(DEFAULT_CONFIG[key])

    DEFAULT_CONFIG['dtype'] = (
        DTYPE_MAP.get(DEFAULT_CONFIG['dtype'], DEFAULT_CONFIG['dtype'])
        if DEFAULT_CONFIG['dtype'] is not None
        else ms.bfloat16
    )

    convert_hf_to_ms(
        input_path=para.huggingface_ckpt_path,
        output_path=para.mindspore_ckpt_path,
        config=DEFAULT_CONFIG
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--huggingface_ckpt_path', default=None, type=str,
                        help="HuggingFace checkpoint directory.")
    parser.add_argument('--mindspore_ckpt_path', default=None, type=str,
                        help="Converted MindSpore Transformers MCore checkpoint directory.")

    parser.add_argument("--num_layers", default=61, type=int,
                        help="The number of attention layers.")
    parser.add_argument("--hidden_size", default=7168, type=int,
                        help="The size of Hidden layer.")
    parser.add_argument("--ffn_hidden_size", default=18432, type=int,
                        help="Transformer Feed-Forward Network hidden size.")
    parser.add_argument("--moe_ffn_hidden_size", default=2048, type=int,
                        help="MoE Feed-Forward Network hidden size.")
    parser.add_argument('--num_routed_experts', default=256, type=int,
                        help="The number of routed experts.")

    parser.add_argument("--num_nextn_predict_layers", default=1, type=int,
                        help="The number of Multi-Token Prediction layers.")
    parser.add_argument("--first_k_dense_replace", default=3, type=int,
                        help="Customizing the number of dense layers.")

    parser.add_argument('--dtype', default='bf16', type=str, choices=['fp16', 'bf16', 'fp32'],
                        help="The dtype of converted weight, choices in ['fp16', 'bf16', 'fp32']")

    args = parser.parse_args()

    convert_weight(args)
