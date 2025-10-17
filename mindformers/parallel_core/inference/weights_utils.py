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
"""weights utils."""
from typing import Any, Dict, Optional
import numpy as np

import mindspore as ms
from mindspore import Parameter

from mindformers.parallel_core.inference.parallel_state import (get_tensor_model_parallel_world_size,
                                                                get_tensor_model_parallel_rank)
from mindformers.version_control import is_310p


def set_weight_attrs(
        weight: Parameter,
        weight_attrs: Optional[Dict[str, Any]],
):
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        setattr(weight, key, value)


def default_weight_loader(param: Parameter, loaded_weight: Any) -> None:
    """Default weight loader."""
    loaded_weight = loaded_weight[:]
    loaded_weight = loaded_weight.astype(np.float16) \
        if (str(loaded_weight.dtype) == 'bfloat16' and is_310p()) else loaded_weight
    param.set_data(ms.Tensor(loaded_weight, dtype=param.dtype))


def split_loaded_weight(loaded_weight, shard_dim, start_idx, shard_size):
    """
    Read numpy slice data based on axis and slice range.
    :loaded_weight: PySafeSlice object
    :shard_dim: axis of weight slice
    :start_idx: start slice index
    :shard_size: end slice index
    """
    if shard_dim is None:
        loaded_weight = loaded_weight[:]
        loaded_weight = loaded_weight.astype(np.float16) \
            if (str(loaded_weight.dtype) == 'bfloat16' and is_310p()) else loaded_weight
        return loaded_weight

    end_idx = start_idx + shard_size
    if shard_dim == 0:
        loaded_weight = loaded_weight[start_idx:end_idx]
    elif shard_dim == 1:
        loaded_weight = loaded_weight[:, start_idx:end_idx]
    elif shard_dim == 2:
        loaded_weight = loaded_weight[:, :, start_idx:end_idx]
    else:
        raise ValueError("shard_dim:{} is not supported.".format(shard_dim))
    loaded_weight = loaded_weight.astype(np.float16) \
        if (str(loaded_weight.dtype) == 'bfloat16' and is_310p()) else loaded_weight
    return loaded_weight


def infer_trans_rope_weight(weight, qk_pos_emb_head_dim):
    """process rope router weight"""
    w1 = weight[..., -qk_pos_emb_head_dim::2, :]
    w2 = weight[..., -qk_pos_emb_head_dim + 1::2, :]
    weight[..., -qk_pos_emb_head_dim:, :] = np.concatenate([w1, w2], axis=-2)
    return weight


def deal_linear_q_up_weight(weight, config, shard_dim, shard_size, rope_transition=False):
    """Splits the linear_q_up weights from source checkpoint.

    Args:
        weight: The weights to be loaded.
        config: Model configuration.
        shard_dim: The dimension of the linear projection after splitting.
        shard_size: The size of each shard when splitting the weight tensor.
                    Used for model parallelism when the model is distributed
                    across multiple devices.
        rope_transition: if true the weights are transformed to fit rope kernel.

    """
    tp_rank = get_tensor_model_parallel_rank()
    num_heads = config.num_attention_heads
    if rope_transition:
        ori_shape_len = len(weight.shape)
        rope_dim = config.qk_pos_emb_head_dim + config.qk_head_dim
        weight = weight.reshape(num_heads, rope_dim, -1)
        weight = infer_trans_rope_weight(weight, config.qk_pos_emb_head_dim)
        weight = weight.reshape(num_heads * rope_dim, -1) if ori_shape_len == 2 else \
                weight.reshape(-1)

    start_idx = tp_rank * shard_size
    loaded_weight = split_loaded_weight(weight, shard_dim, start_idx, shard_size)
    if loaded_weight.shape[-1] == 1 and len(loaded_weight.shape) == 2:
        loaded_weight = loaded_weight.reshape(-1)
    return loaded_weight


def deal_linear_kv_up_weight(weight, config, shard_dim, shard_size):
    """Splits the linear_kv_up weights from source checkpoint.

    Args:
        weight: The weights to be loaded.
        config: Model configuration.
        shard_dim: The dimension of the linear projection after splitting.
        shard_size: The size of each shard when splitting the weight tensor.
                    Used for model parallelism when the model is distributed
                    across multiple devices.

    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    qk_nope_head_dim = config.qk_head_dim
    v_head_dim = config.v_head_dim
    lkv2kv_head = qk_nope_head_dim + v_head_dim
    num_heads = config.num_attention_heads

    k_shard_size = (num_heads * qk_nope_head_dim) // tp_size
    v_shard_size = (num_heads * v_head_dim) // tp_size
    kv_shard_size = k_shard_size + v_shard_size
    if kv_shard_size != shard_size:
        raise ValueError(f'The sum of k_shard_size and v_shard_size should equal shard_size, '
                         f'but currently the k_shard_size is {k_shard_size} and the v_shard_size is {v_shard_size}, '
                         f'the shard_size is {shard_size}, '
                         f'the sum of k_shard_size and v_shard_size does not equal shard_size.')

    weight = weight.reshape(num_heads, lkv2kv_head, -1)
    value_k_nope, value_v = (weight[:, :qk_nope_head_dim, :],
                             weight[:, qk_nope_head_dim:, :])
    # value_k_nope
    value_k_nope = value_k_nope.reshape(-1, value_k_nope.shape[-1])
    k_start_idx = tp_rank * k_shard_size
    value_k_nope = split_loaded_weight(value_k_nope, shard_dim, k_start_idx, k_shard_size)
    # value_v_nope
    value_v = value_v.reshape(-1, value_v.shape[-1])
    v_start_idx = tp_rank * v_shard_size
    value_v = split_loaded_weight(value_v, shard_dim, v_start_idx, v_shard_size)
    weight = np.concatenate((value_k_nope, value_v), 0)
    return weight


def deal_linear_kv_down_weight(weight, config, rope_transition=False):
    """Splits the linear_kv_down weights from source checkpoint.

    Args:
        weight: The weights to be loaded.
        config: Model configuration.
        rope_transition: if true the weights are transformed to fit rope kernel

    """
    if rope_transition:
        kv_lora_rank = config.kv_lora_rank
        qk_rope_head_dim = config.qk_pos_emb_head_dim
        kv_head_dim = kv_lora_rank + qk_rope_head_dim
        ori_shape_len = len(weight.shape)
        weight = weight.reshape(kv_head_dim, -1)
        weight = infer_trans_rope_weight(weight, qk_rope_head_dim)
        if ori_shape_len == 1:
            weight = weight.reshape(-1)
    return weight


def deal_training_qkv_weight(weight, config):
    """
    Process QKV weights during training by splitting and reorganizing according to tensor parallel strategy.

    Args:
        weight (np.ndarray): Original QKV weight matrix, shape may be 1D or 2D.
        config (object): Model configuration object containing kv_channels, hidden_size,
                        num_attention_heads, and num_query_groups attributes.

    Returns:
        np.ndarray: QKV weights after tensor parallel splitting, shape corresponds to input.
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    head_dim = config.kv_channels if config.kv_channels else config.hidden_size // config.num_attention_heads
    q_channel = config.num_attention_heads * head_dim
    kv_channel = config.num_query_groups * head_dim
    n_kv_heads = config.num_query_groups or config.num_attention_heads
    n_rep = config.num_attention_heads // n_kv_heads

    qkv_dim = len(weight.shape)
    if qkv_dim == 1:
        w = weight.shape[0]
        weight = weight.reshape(w, -1)
    weight = weight.reshape(n_kv_heads, (n_rep + 2) * head_dim, -1)
    q_weight = weight[:, :n_rep * head_dim, :]
    k_weight = weight[:, n_rep * head_dim:n_rep * head_dim + head_dim, :]
    v_weight = weight[:, n_rep * head_dim + head_dim:n_rep * head_dim + 2 * head_dim, :]
    q_weight = q_weight.reshape(q_channel, -1)
    k_weight = k_weight.reshape(kv_channel, -1)
    v_weight = v_weight.reshape(kv_channel, -1)

    q_shard_size = q_channel // tp_size
    q_start_idx = tp_rank * q_shard_size
    q_weight = split_loaded_weight(q_weight, 0, q_start_idx, q_shard_size)
    if tp_size > config.num_query_groups:
        replicate = tp_size // config.num_query_groups
        kv_shard_size = kv_channel // (tp_size // replicate)
        kv_start_idx = tp_rank // replicate * kv_shard_size
    else:
        kv_shard_size = kv_channel // tp_size
        kv_start_idx = tp_rank * kv_shard_size
    k_weight = split_loaded_weight(k_weight, 0, kv_start_idx, kv_shard_size)
    v_weight = split_loaded_weight(v_weight, 0, kv_start_idx, kv_shard_size)
    cat_qkv_weight = np.concatenate((q_weight, k_weight, v_weight), axis=0)
    if qkv_dim == 1:
        cat_qkv_weight = cat_qkv_weight.reshape(w // tp_size,)
    return cat_qkv_weight


def deal_training_ffn_weight(weight, config):
    """
    Process FFN layer weights during training by splitting and reorganizing according to tensor parallel strategy.

    This function splits FFN layer weights according to tensor parallel strategy,
    supporting both 1D and 2D weight formats, and returns W1 and W3 weights concatenated after splitting.

    Args:
        weight (np.ndarray): FFN layer weight matrix, can be 1-dimensional or 2-dimensional array
        config (object): Model configuration object that needs to contain ffn_hidden_size attribute

    Returns:
        np.ndarray: FFN weights after tensor parallel splitting, with W1 and W3 weights concatenated along axis 0
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    ffn_hidden_size = config.ffn_hidden_size

    ffn_dim = len(weight.shape)
    if ffn_dim == 1:
        w = weight.shape[0]
        weight = weight.reshape(w, -1)

    weight = weight.reshape(ffn_hidden_size, 2, -1)
    w1_weight = weight[:, :1, :]
    w3_weight = weight[:, 1:2, :]
    w1_weight = w1_weight.reshape(ffn_hidden_size, -1)
    w3_weight = w3_weight.reshape(ffn_hidden_size, -1)

    w1_shard_size = w1_weight.shape[0] // tp_size
    w1_start_idx = tp_rank * w1_shard_size
    w1_weight = split_loaded_weight(w1_weight, 0, w1_start_idx, w1_shard_size)
    w3_shard_size = w3_weight.shape[0] // tp_size
    w3_start_idx = tp_rank * w3_shard_size
    w3_weight = split_loaded_weight(w3_weight, 0, w3_start_idx, w3_shard_size)
    cat_ffn_weight = np.concatenate((w1_weight, w3_weight), axis=0)
    if ffn_dim == 1:
        cat_ffn_weight = cat_ffn_weight.reshape(w // tp_size,)
    return cat_ffn_weight


def deal_training_moe_weight(weight):
    """
    Process weights for MoE model training by splitting and reorganizing.

    This function is mainly used to split MoE layer weights according to tensor parallelism
    during model parallel training, and merge W1 and W3 weights before returning.

    Args:
        weight: Input weight tensor, usually containing combined W1 and W3 weights

    Returns:
        cat_moe_weight: Weight tensor after splitting and merging according to tensor parallelism
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    w = weight.shape[1]
    w1_weight = weight[:, : w // 2]
    w1_shard_size = w1_weight.shape[1] // tp_size
    w1_start_idx = tp_rank * w1_shard_size
    w1_weight = split_loaded_weight(w1_weight, 1, w1_start_idx, w1_shard_size)
    w3_weight = weight[:, w // 2: w // 2 * 2]
    w3_shard_size = w3_weight.shape[1] // tp_size
    w3_start_idx = tp_rank * w3_shard_size
    w3_weight = split_loaded_weight(w3_weight, 1, w3_start_idx, w3_shard_size)
    cat_moe_weight = np.concatenate((w1_weight, w3_weight), axis=1)
    return cat_moe_weight


def make_expert_params_mapping(
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,) -> list[tuple[str, str, int, str]]:
    """
    Generate an expert parameter mapping list for mapping expert weights from checkpoints to model parameters.

    Args:
        ckpt_gate_proj_name: Weight name of the gate projection layer in the checkpoint.
        ckpt_down_proj_name: Weight name of the down projection layer in the checkpoint.
        ckpt_up_proj_name: Weight name of the up projection layer in the checkpoint.
        num_experts: Number of experts

    Returns:
        A list of expert parameter mappings, where each element is a tuple of
        (param_name, weight_name, expert_id, shard_id).
        - param_name: Prefix of the model parameter name
        - weight_name: Weight name of the logical expert
        - expert_id: Physical expert ID
        - shard_id: Shard ID
    """
    # In the returned mapping:
    # - `expert_id` is the physical expert id
    # - `weight_name` contains the weight name of the logical expert
    # So that we should map the expert id to logical in `weight_name`

    params_mapping = []
    expert_list = [('w1', ckpt_gate_proj_name), ('w2', ckpt_down_proj_name), ('w3', ckpt_up_proj_name)]

    for shard_id, shard_weight_name in expert_list:
        is_linear_fc1 = shard_weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
        weight_param_name = f'experts.{"weight1" if is_linear_fc1 else "weight2"}'
        other_param_name = f'experts.{"linear_fc1" if is_linear_fc1 else "linear_fc2"}.'
        for expert_id in range(num_experts):
            weight_mapping = (
                weight_param_name,
                f'experts.{expert_id}.{shard_weight_name}.weight',
                expert_id,
                shard_id
            )
            other_param_mapping = (
                other_param_name,
                f'experts.{expert_id}.{shard_weight_name}.',
                expert_id,
                shard_id
            )
            params_mapping.append(weight_mapping)
            params_mapping.append(other_param_mapping)
    return params_mapping


def make_expert_params_mapping_with_expert_dim(
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str) -> list[tuple[str, str, str]]:
    """
    Generate an expert parameter mapping list for mapping expert weights from checkpoints to model parameters.

    Args:
        ckpt_gate_proj_name: Weight name of the gate projection layer in the checkpoint.
        ckpt_down_proj_name: Weight name of the down projection layer in the checkpoint.
        ckpt_up_proj_name: Weight name of the up projection layer in the checkpoint.

    Returns:
        A list of expert parameter mappings, where each element is a tuple of
        (param_name, weight_name, shard_id).
        - param_name: Prefix of the model parameter name
        - weight_name: Weight name of the logical expert
        - shard_id: Shard ID
    """
    weight_params_mapping = []
    shard_map = [('w1', ckpt_gate_proj_name), ('w2', ckpt_down_proj_name), ('w3', ckpt_up_proj_name)]
    for shard_id, weight_name in shard_map:
        if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]:
            weight1_prefix = "experts.weight1"
        else:
            weight1_prefix = "experts.weight2"
        param_tuple = (
            weight1_prefix,
            f"experts.{weight_name}.weight",
            shard_id
        )
        weight_params_mapping.append(param_tuple)
    return weight_params_mapping


def split_fusion_loaded_weight(loaded_weight, start_idxs, shard_sizes):
    """
    Read numpy slice data based on axis0 and slice range.
    loaded_weight: PySafeSlice object
    start_idxs: weight shard start slice indexes
    shard_sizes: weight shard sizes
    """
    loaded_weight_parts = []
    for start_idx, shard_size in zip(start_idxs, shard_sizes):
        loaded_weight_parts.append(loaded_weight[start_idx:start_idx + shard_size])
    perrank_ffn_weight = np.concatenate(loaded_weight_parts, axis=0)
    return perrank_ffn_weight
