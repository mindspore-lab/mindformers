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
""" utils """
__all__ = [
    "get_attn_mask_func",
    "generate_state_dict",
    "generate_padding_index",
    "update_comm_config",
]

from contextlib import contextmanager
import numpy as np

import mindspore as ms
from mindspore import Tensor, ops, Parameter, mint
from mindspore.communication import get_group_size, get_rank

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.parallel_state import (get_data_parallel_group,
                                                                get_tensor_model_parallel_world_size,
                                                                get_data_parallel_world_size,
                                                                get_moe_expert_parallel_world_size,
                                                                get_moe_tensor_parallel_world_size)
from mindformers.tools import logger


def attn_mask_fill(attention_scores: Tensor, attention_mask, fill_value=-10000.0):
    """mask attention scores with the mask value"""
    attention_scores = ops.masked_fill(
        attention_scores,
        attention_mask,
        Tensor(fill_value, attention_scores.dtype),
    )
    return attention_scores


def attn_mask_add(attention_scores: Tensor, attention_mask):
    """Llama attention mask function"""
    score_dtype = attention_scores.dtype
    attention_scores = ops.add(
        attention_scores, ops.Cast()(attention_mask, score_dtype)
    )
    return attention_scores


ATTNMASK_FUNC_MAP = {
    "attn_mask_fill": attn_mask_fill,
    "attn_mask_add": attn_mask_add,
}


def get_attn_mask_func(mask_func_type):
    r"""
    Get attention mask function.

    Args:
        mask_func_type (str): The attention mask function type.

    Returns:
        Function, the attention mask function.
    """
    if mask_func_type not in ATTNMASK_FUNC_MAP:
        raise KeyError("Invalid attention mask function. Supported attention "
                       "mask function are ['attn_mask_fill', 'attn_mask_add'] "
                       ", but got {}.".format(mask_func_type))
    return ATTNMASK_FUNC_MAP[mask_func_type]


def _update_sharded_state_dict(network, sharded_state_dict):
    """Update shared state dict with network"""
    cells = network.name_cells()
    for _, subcell in cells.items():
        if subcell == network:
            continue
        if hasattr(subcell, "sharded_state_dict"):
            sharded_state_dict.update(subcell.sharded_state_dict())
        else:
            _update_sharded_state_dict(subcell, sharded_state_dict)


def generate_state_dict(network):
    """Generate the sharded state dict for network"""

    state_dict = {
        "total_rank": get_group_size(),
        "stage_rank_size": get_group_size(),
        "stage": 0
    }
    model_state_dict = {}
    _update_sharded_state_dict(network=network, sharded_state_dict=model_state_dict)
    model_param_dict = network.parameters_dict()

    for name in model_param_dict:
        if name not in model_state_dict:
            model_state_dict[name] = {'shape': model_param_dict[name].shape,
                                      'shard': tuple([1] * model_param_dict[name].ndim)}

    state_dict['model'] = model_state_dict
    state_dict['optimizer'] = {}
    return state_dict


def get_tp_world_size():
    tp_size = get_tensor_model_parallel_world_size()
    return tp_size if tp_size else 1


def get_moe_tp_world_size():
    moe_tp_size = get_moe_tensor_parallel_world_size()
    return moe_tp_size if moe_tp_size else 1


def get_moe_ep_world_size():
    moe_ep_size = get_moe_expert_parallel_world_size()
    return moe_ep_size if moe_ep_size else 1


def get_dp_world_size():
    dp_size = get_data_parallel_world_size()
    return dp_size if dp_size else 1


def create_empty_parameter(shape, *, dtype=None, device=None, **kwargs):
    """Create an empty parameter."""
    def get_param(*args):
        return [Tensor, args[0]]

    @contextmanager
    def replace_class_method(cls, name, new_method):
        old_method = getattr(cls, name)

        setattr(cls, name, new_method)
        yield
        setattr(cls, name, old_method)

    data = mint.empty(shape, dtype=dtype, device=device)

    with replace_class_method(Parameter, "_get_parameter_new_args", get_param):
        param = Parameter(data, **kwargs)
    return param


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    if numerator % denominator != 0:
        raise ValueError("{} is not divisible by {}".format(numerator, denominator))


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def save_strategy_file(state_dict, strategy_file_name):
    r"""
    Save the strategy file according to the state_dict and strategy_file_name

    Args:
        state_dict (Dict): dict with sharding metainfo
        strategy_file_name (String): the name of the target saving file

    Supported Platforms:
        ``Ascend``
    """
    import os
    import stat
    from mindspore.train.node_strategy_pb2 import ParallelStrategyMap as ckpt_strategy

    stra = ckpt_strategy()

    stage_rank_size = state_dict["stage_rank_size"]
    stage = state_dict["stage"]
    model_param = state_dict["model"]
    optimizer_param = state_dict["optimizer"]
    stra.current_stage = 0
    model_param.update(optimizer_param)
    for name, item in model_param.items():
        if "shard" not in item or "shape" not in item:
            continue
        opt_weight_shard_step = item["opt_weight_shard_step"] if "opt_weight_shard_step" in item.keys() else 0
        opt_weight_shard_size = item["opt_weight_shard_size"] if "opt_weight_shard_size" in item.keys() else 0
        strategy_item = stra.parallel_strategy_item.add()
        strategy_item.node_name = name
        parallel_strategys = strategy_item.parallel_strategys
        parallel_strategys.stage = stage
        shard = item["shard"]
        shape = item["shape"]
        parallel_strategy = parallel_strategys.parallel_strategy.add()
        shard_mul = 1
        for ele in shard:
            parallel_strategy.dim.append(ele)
            shard_mul = shard_mul * ele
        layout_item = stra.parallel_layout_item.add()
        layout_item.param_name = name
        parallel_layouts = layout_item.parallel_layouts
        parallel_layouts.field = 0
        parallel_layouts.opt_weight_shard_step = opt_weight_shard_step
        parallel_layouts.opt_weight_shard_size = opt_weight_shard_size
        dev_matrix = parallel_layouts.dev_matrix.add()
        if stage_rank_size == shard_mul:
            repeat_calc_num = 1
        elif stage_rank_size % shard_mul == 0:
            repeat_calc_num = stage_rank_size // shard_mul
        else:
            raise ValueError(
                f"For {name}, the shard{shard} requires {shard_mul} devices, "
                f"but the device number of this stage is {stage_rank_size}, "
                f"it can not be divisible by {shard_mul}"
            )
        if repeat_calc_num != 1:
            dev_matrix.dim.append(repeat_calc_num)
        for ele in shard:
            dev_matrix.dim.append(ele)
        tensor_map = parallel_layouts.tensor_map.add()
        shape_len = len(shape)
        index = shape_len - 1
        for _ in range(shape_len):
            tensor_map.dim.append(index)
            index = index - 1
        param_split_shape = parallel_layouts.param_split_shape.add()
        for ele in shape:
            param_split_shape.dim.append(ele)

    try:
        if os.path.exists(strategy_file_name):
            os.chmod(strategy_file_name, stat.S_IWUSR)
        if "/" in strategy_file_name:
            real_path = os.path.abspath(strategy_file_name[: strategy_file_name.rfind("/")])
            os.makedirs(real_path, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(strategy_file_name, flags, modes), 'wb') as f:
            f.write(stra.SerializeToString())
            os.chmod(strategy_file_name, stat.S_IRUSR)

    except BaseException as e:
        logger.critical(
            f"Failed to save the checkpoint file {strategy_file_name}. Maybe don't have "
            f"the permission to write files, or the disk space is insufficient and so on."
        )
        raise e


def generate_padding_index(q_seq_lens: Tensor):
    """generate padding index parameter in TP Region."""
    tp_group_size = get_tensor_model_parallel_world_size()
    tokens_len_per_dp = q_seq_lens.sum().reshape(-1)
    tokens_len_per_dp = ops.AllGather(group=get_data_parallel_group().group)(tokens_len_per_dp)
    tokens_len_per_dp = tokens_len_per_dp.asnumpy()
    padding_size = (tokens_len_per_dp.max() + tp_group_size - 1) // tp_group_size * tp_group_size
    current_dp_rank = get_rank() // tp_group_size
    attn_padding_idx = None
    attn_unpadding_idx = None
    ffn_padding_idx = None
    ffn_unpadding_idx = None
    last_arange_index = 0

    for dp_rank, tokens_length in enumerate(tokens_len_per_dp):
        arange_data = np.arange(0, int(tokens_length), dtype=np.int32)
        if dp_rank == current_dp_rank:
            ffn_unpadding_idx = arange_data
            pad = np.zeros(padding_size - arange_data.shape[0], dtype=np.int32)
            attn_padding_idx = np.concatenate((arange_data, pad), axis=0)
        if dp_rank == 0:
            attn_unpadding_idx = arange_data
            last_arange_index = arange_data[-1]
            pad = np.zeros(padding_size - attn_unpadding_idx.shape[0], dtype=np.int32)
            ffn_padding_idx = np.concatenate((attn_unpadding_idx, pad), axis=0)
        else:
            attn_offset_idx = arange_data + padding_size * dp_rank
            attn_unpadding_idx = np.concatenate((attn_unpadding_idx, attn_offset_idx), axis=0)
            ffn_offset_idx = arange_data + last_arange_index + 1
            last_arange_index = ffn_offset_idx[-1]
            pad = np.zeros(padding_size - ffn_offset_idx.shape[0], dtype=np.int32)
            ffn_padding_idx = np.concatenate((ffn_padding_idx, ffn_offset_idx, pad), axis=0)

    attn_padding_idx = ms.from_numpy(attn_padding_idx)
    attn_unpadding_idx = ms.from_numpy(attn_unpadding_idx)
    ffn_padding_idx = ms.from_numpy(ffn_padding_idx)
    ffn_unpadding_idx = ms.from_numpy(ffn_unpadding_idx)
    return attn_padding_idx, attn_unpadding_idx, ffn_padding_idx, ffn_unpadding_idx


def update_comm_config(config: TransformerConfig):
    """update communication config"""
    global_group_size = get_group_size()
    tp_group_size = config.tensor_model_parallel_size
    dp_group_size = global_group_size // tp_group_size
    ep_group_size = config.expert_model_parallel_size
    moe_tp_group_size = global_group_size // ep_group_size

    if dp_group_size > 1 and tp_group_size == 1:
        if moe_tp_group_size == 1:
            config.attn_allreduce = False
            config.ffn_allreduce = False
            config.use_alltoall = True
        else:
            config.attn_allgather = True
            config.attn_allreduce = False
            config.ffn_reduce_scatter = True
            config.ffn_allreduce = False
    elif dp_group_size > 1:
        if moe_tp_group_size == 1:
            config.attn_reduce_scatter = True
            config.attn_allreduce = False
            config.ffn_allgather = True
            config.ffn_allreduce = False
            config.use_alltoall = True
        else:
            config.attn_reduce_scatter = True
            config.attn_allgather = True
            config.attn_allreduce = False
            config.ffn_reduce_scatter = True
            config.ffn_allgather = True
            config.ffn_allreduce = False
    return config
