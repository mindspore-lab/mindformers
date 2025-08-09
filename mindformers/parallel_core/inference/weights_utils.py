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
    return loaded_weight


def infer_trans_rope_weight(weight, qk_pos_emb_head_dim):
    """process rope router weight"""
    w1 = weight[..., -qk_pos_emb_head_dim::2, :]
    w2 = weight[..., -qk_pos_emb_head_dim + 1::2, :]
    weight[..., -qk_pos_emb_head_dim:, :] = np.concatenate([w1, w2], axis=-2)
    return weight


def deal_linear_q_up_weight(weight, config, shard_dim, shard_size):
    """Splits the linear_q_up weights from source checkpoint.

    Args:
        weight: The weights to be loaded.
        config: Model configuration.
        shard_dim: The dimension of the linear projection after splitting.
        shard_size: The size of each shard when splitting the weight tensor.
                    Used for model parallelism when the model is distributed
                    across multiple devices.

    """
    tp_rank = get_tensor_model_parallel_rank()
    num_heads = config.num_attention_heads
    rope_dim = config.qk_pos_emb_head_dim + config.qk_head_dim
    weight = weight.reshape(num_heads, rope_dim, -1)
    weight = infer_trans_rope_weight(weight, config.qk_pos_emb_head_dim)
    weight = weight.reshape(num_heads * rope_dim, -1)
    start_idx = tp_rank * shard_size
    loaded_weight = split_loaded_weight(weight, shard_dim, start_idx, shard_size)
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


def deal_linear_kv_down_weight(weight, config):
    kv_lora_rank = config.kv_lora_rank
    qk_rope_head_dim = config.qk_pos_emb_head_dim
    kv_head_dim = kv_lora_rank + qk_rope_head_dim
    weight = weight.reshape(kv_head_dim, -1)
    weight = infer_trans_rope_weight(weight, qk_rope_head_dim)
    return weight


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

    return [
        # (param_name, weight_name, expert_id, shard_id)
        ("experts.weight1." if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.weight2.",
         f"experts.{expert_id}.{weight_name}.",
         expert_id, shard_id) for expert_id in range(num_experts)
        for shard_id, weight_name in [
            ('w1', ckpt_gate_proj_name),
            ('w2', ckpt_down_proj_name),
            ('w3', ckpt_up_proj_name),
        ]
    ]
