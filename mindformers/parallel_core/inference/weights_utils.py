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
"""attention and qkv concat."""
from typing import Any, Dict, Optional
from safetensors import safe_open
import numpy as np

import mindspore as ms
from mindspore import Parameter
from mindspore.communication.management import get_rank

from mindformers.parallel_core.inference.utils import get_tp_world_size


file_handles = {}


def set_weight_attrs(
        weight: Parameter,
        weight_attrs: Optional[Dict[str, Any]],
):
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        setattr(weight, key, value)


class WeightsLoader:

    """A utility class for handling model weights (loading, converting, processing)."""

    def __init__(self, weights_path):
        self.tp_group_size = get_tp_world_size()
        self.global_rank_id = get_rank()
        self.tp_rank_id = self.global_rank_id % self.tp_group_size
        self.mf_hf_mapping = {}
        self.mapping_dict = {}
        self.parameter_dict = {}
        self.weights_path = weights_path

    def not_split(self, src_keys_dict, net_name, config):
        for weight_name, file in src_keys_dict.items():
            weight_value = get_file_handles(f'{self.weights_path}/{file}').get_tensor(weight_name)
            if weight_name.split(".")[-1] == "e_score_correction_bias":
                self.parameter_dict[net_name] = ms.Parameter(ms.from_numpy(weight_value),
                                                             name=net_name, requires_grad=False)
            else:
                self.parameter_dict[net_name] = ms.Parameter(
                    ms.from_numpy(weight_value).astype(getattr(config, 'params_dtype')),
                    name=net_name, requires_grad=False)

    def split_by_tp_rank_rows(self, src_keys_dict, net_name, config):
        for weight_name, file in src_keys_dict.items():
            weight_value = get_file_handles(f'{self.weights_path}/{file}').get_slice(weight_name)
            split_data = split_weight_by_tp_rank(
                weight_value, split_axis=1, tp_rank_id=self.tp_rank_id, tp_group_size=self.tp_group_size)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(split_data).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)

    def split_by_tp_rank_columns(self, src_keys_dict, net_name, config):
        for weight_name, file in src_keys_dict.items():
            weight_value = get_file_handles(f'{self.weights_path}/{file}').get_slice(weight_name)
            split_data = split_weight_by_tp_rank(
                weight_value, split_axis=0, tp_rank_id=self.tp_rank_id, tp_group_size=self.tp_group_size)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(split_data).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)

    def add_qkv_ffn_weight_into_dict(self, src_keys_dict, net_name, config):
        """Splits and concat the  QKV or FFN or shared experts weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            config: Model configuration.

        """
        if len(src_keys_dict) > 1:
            if len(src_keys_dict) == 3:
                weight_mapping = {
                    self.mf_hf_mapping['linear_q']: 0,
                    self.mf_hf_mapping['linear_k']: None,
                    self.mf_hf_mapping['linear_v']: None
                }
            elif len(src_keys_dict) == 2:
                weight_mapping = {
                    self.mf_hf_mapping['gating']: 0,
                    self.mf_hf_mapping['linear_fc1']: 0,
                }
            else:
                raise ValueError(f'There should be three key values for attention weights or '
                                 f'two values for mlp weights but {len(src_keys_dict)} are present')
            weight_value = deal_concat_weight(
                config, src_keys_dict, weight_mapping, self.weights_path, self.tp_rank_id, self.tp_group_size)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(weight_value).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)
        else:
            for weight_name, file in src_keys_dict.items():
                weight_value = get_file_handles(f'{self.weights_path}/{file}').get_tensor(weight_name)
                if self.mf_hf_mapping.get('linear_qkv') == weight_name.split('.')[-2]:
                    weight_value = handle_training_qkv_weight(weight_value, config)
                else:
                    weight_value = handle_training_ffn_weight(weight_value)
                self.parameter_dict[net_name] = ms.Parameter(
                    ms.from_numpy(weight_value).astype(getattr(config, 'params_dtype')),
                    name=net_name, requires_grad=False)

    def add_router_expert_weight1_into_dict(self, src_keys_dict, net_name, config):
        """Splits and concat the router gating, linear_fc1 of weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            config: Model configuration.

        """
        router_gate_dict = {}
        router_up_dict = {}
        for weight_name, file in src_keys_dict.items():
            if self.mf_hf_mapping['gating'] in weight_name:
                router_gate_dict.update({weight_name: file})
            else:
                router_up_dict.update({weight_name: file})
        router_gate_value = deal_router_expert_weight(
            router_gate_dict, self.weights_path, self.tp_rank_id, self.tp_group_size, split_axis=0)
        router_up_value = deal_router_expert_weight(
            router_up_dict, self.weights_path, self.tp_rank_id, self.tp_group_size, split_axis=0)
        linear_fc1_value = np.concatenate([router_gate_value, router_up_value], axis=1)
        linear_fc1_value = ms.from_numpy(linear_fc1_value).permute(
            0, 2, 1).astype(dtype=getattr(config, 'params_dtype'))
        self.parameter_dict[net_name] = ms.Parameter(linear_fc1_value, name=net_name, requires_grad=False)

    def add_router_expert_weight2_into_dict(self, src_keys_dict, net_name, config):
        """Splits the router linear_fc2 weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            config: Model configuration.

        """
        w2_value_all = deal_router_expert_weight(
            src_keys_dict, self.weights_path, self.tp_rank_id, self.tp_group_size, split_axis=1)
        linear_fc2_value = ms.from_numpy(w2_value_all).permute(
            0, 2, 1).astype(dtype=getattr(config, 'params_dtype'))
        self.parameter_dict[net_name] = ms.Parameter(linear_fc2_value, name=net_name, requires_grad=False)

    def add_linear_kv_down_proj_into_dict(self, src_keys_dict, net_name, config):
        """Splits or concat linear_q_down or linear_kv_down weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            config: Model configuration.

        """
        q_down_value, kv_down_value = None, None
        for weight_name, file in src_keys_dict.items():
            if self.mf_hf_mapping.get('linear_q_down_proj') == weight_name.split('.')[-2]:
                q_down_value = get_file_handles(f'{self.weights_path}/{file}').get_tensor(weight_name)
            if self.mf_hf_mapping.get('linear_kv_down_proj') == weight_name.split('.')[-2]:
                kv_down_value = deal_linear_kv_down_weight(config, weight_name, file, self.weights_path)
        if kv_down_value is not None:
            if q_down_value is not None:
                linear_qkv_down_proj_value = np.concatenate((q_down_value, kv_down_value), 0)
                self.parameter_dict[net_name] = ms.Parameter(
                    ms.from_numpy(linear_qkv_down_proj_value).astype(getattr(config, 'params_dtype')),
                    name=net_name, requires_grad=False)
            else:
                self.parameter_dict[net_name] = ms.Parameter(
                    ms.from_numpy(kv_down_value).astype(getattr(config, 'params_dtype')),
                    name=net_name, requires_grad=False)
        else:
            raise ValueError(
                'The weight files are missing proper linear_kv_down_proj weight matrices.')

    def add_linear_q_up_proj_into_dict(self, src_keys_dict, net_name, config):
        """Splits the linear_q_up weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            config: Model configuration.

        """
        for weight_name, file in src_keys_dict.items():
            linear_q_up_proj_value = get_file_handles(f'{self.weights_path}/{file}').get_tensor(weight_name)
            num_heads = config.num_attention_heads
            rope_dim = config.qk_pos_emb_head_dim + config.qk_head_dim
            linear_q_up_proj_value = linear_q_up_proj_value.reshape(num_heads, rope_dim, -1)
            linear_q_up_proj_value = infer_trans_rope_weight(linear_q_up_proj_value, config.qk_pos_emb_head_dim)
            linear_q_up_proj_value = linear_q_up_proj_value.reshape(num_heads * rope_dim, -1)
            linear_q_up_proj_value = split_weight_by_tp_rank(
                linear_q_up_proj_value, split_axis=0, tp_rank_id=self.tp_rank_id, tp_group_size=self.tp_group_size)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(linear_q_up_proj_value).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)

    def add_linear_kv_up_proj_into_dict(self, src_keys_dict, net_name, config):
        """Splits the linear_kv_up weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            config: Model configuration.

        """
        for weight_name, file in src_keys_dict.items():
            linear_kv_up_proj_value = get_file_handles(f'{self.weights_path}/{file}').get_tensor(weight_name)
            qk_nope_head_dim = config.qk_head_dim
            v_head_dim = config.v_head_dim
            lkv2kv_head = qk_nope_head_dim + v_head_dim
            num_heads = config.num_attention_heads
            linear_kv_up_proj_value = linear_kv_up_proj_value.reshape(num_heads, lkv2kv_head, -1)
            value_k_nope, value_v = (linear_kv_up_proj_value[:, :qk_nope_head_dim, :],
                                     linear_kv_up_proj_value[:, qk_nope_head_dim:, :])
            # value_k_nope
            value_k_nope = value_k_nope.reshape(-1, value_k_nope.shape[-1])
            value_k_nope = split_weight_by_tp_rank(
                value_k_nope, split_axis=0, tp_rank_id=self.tp_rank_id, tp_group_size=self.tp_group_size)
            # value_v_nope
            value_v = value_v.reshape(-1, value_v.shape[-1])
            value_v = split_weight_by_tp_rank(
                value_v, split_axis=0, tp_rank_id=self.tp_rank_id, tp_group_size=self.tp_group_size)
            linear_kv_up_proj_value = np.concatenate((value_k_nope, value_v), 0)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(linear_kv_up_proj_value).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)


def get_file_handles(filename):
    """Get cached file handle or create new one if not exists.

    Args:
        filename (str): Path to the file to open

    Returns:
        FileHandle: File handle object (via safe_open)
    """
    if filename not in file_handles:
        fp = safe_open(filename, framework="np")
        file_handles[filename] = fp
    return file_handles[filename]


def infer_trans_rope_weight(weight, qk_pos_emb_head_dim):
    """process rope router weight"""
    w1 = weight[..., -qk_pos_emb_head_dim::2, :]
    w2 = weight[..., -qk_pos_emb_head_dim + 1::2, :]
    weight[..., -qk_pos_emb_head_dim:, :] = np.concatenate([w1, w2], axis=-2)
    return weight


def split_weight_by_tp_rank(weight, split_axis, tp_rank_id, tp_group_size):
    """
    Split model weight by the current rank id for tensor parallelism.

    Args:
        weight(np.array): The full weight tensor to be split.
        split_axis(int): The axis along which to split the weight. Usually 0 (row-wise) or 1 (column-wise).
        tp_rank_id:
        tp_group_size:

    Returns:
        weight: The split sub-tensor assigned to the current rank.
    """

    if isinstance(weight, np.ndarray):
        shape = weight.shape
    else:
        shape = weight.get_shape()
    split_size = shape[split_axis] // tp_group_size
    start = tp_rank_id * split_size
    stop = (tp_rank_id + 1) * split_size
    return weight[start:stop] if split_axis == 0 else weight[:, start:stop]


def deal_linear_kv_down_weight(config, weight_name, file, weights_path):
    linear_kv_down_value = get_file_handles(
        f'{weights_path}/{file}').get_tensor(weight_name)
    kv_lora_rank = config.kv_lora_rank
    qk_rope_head_dim = config.qk_pos_emb_head_dim
    kv_head_dim = kv_lora_rank + qk_rope_head_dim
    linear_kv_down_value = linear_kv_down_value.reshape(kv_head_dim, -1)
    linear_kv_down_value = infer_trans_rope_weight(linear_kv_down_value, qk_rope_head_dim)
    return linear_kv_down_value


def deal_router_expert_weight(src_keys_dict, weights_path, tp_rank_id, tp_group_size, split_axis):
    """Process and combine expert weights for model parallel (tensor parallel) training.

    Args:
        src_keys_dict (dict): Dictionary mapping weight names to their file paths. Format: {weight_name: file_path}
        weights_path (str): Base directory path where weight files are stored.
        tp_rank_id (int): Tensor parallel rank ID of the current process.
        tp_group_size (int): Total number of devices in the tensor parallel group.
        split_axis (int): Axis along which to split the weight tensor for model parallelism.

    Returns:
        numpy.ndarray: Combined and properly split weight tensor for the current rank.
    """
    weight_list = []
    sorted_items = sorted(
        src_keys_dict.items(),
        key=lambda x: int(x[0].split('.')[-3])
    )
    src_keys_dict = {}
    for k, v in sorted_items:
        src_keys_dict[k] = v
    for weight_name, file in src_keys_dict.items():
        weight_value = get_file_handles(f'{weights_path}/{file}').get_slice(weight_name)
        weight_value = split_weight_by_tp_rank(
            weight_value, split_axis, tp_rank_id=tp_rank_id, tp_group_size=tp_group_size)
        weight_list.append(weight_value)
    weight_value_all = np.stack(weight_list, axis=0)
    return weight_value_all


def deal_concat_weight(config, src_keys_dict, weight_mapping, weights_path, tp_rank_id, tp_group_size):
    """Process and concatenate weights with tensor parallel (model parallel) support.

    Args:
        config (object): Model configuration object containing parameters.
        src_keys_dict (dict): Dictionary mapping weight names to their file paths. Format: {weight_name: file_path}
        weight_mapping (dict): Dictionary specifying how to split each weight component.
        weights_path (str): Base directory path where weight files are stored.
        tp_rank_id (int): Tensor parallel rank ID of the current process.
        tp_group_size (int): Total number of devices in the tensor parallel group.

    Returns:
           numpy.ndarray: Concatenated weight tensor after proper splitting for the current rank
    """
    results = {}
    for weight_name, file in src_keys_dict.items():
        key = weight_name.split('.')[-2]
        if key in weight_mapping:
            split_axis = weight_mapping[key]
            weight_value = get_file_handles(f'{weights_path}/{file}').get_slice(weight_name)
            if split_axis is None:
                if tp_group_size > config.num_query_groups:
                    replicate = tp_group_size // config.num_query_groups
                    weight_value = split_weight_by_tp_rank(
                        weight_value, split_axis=0, tp_rank_id=tp_rank_id // replicate,
                        tp_group_size=config.num_query_groups
                    )
                else:
                    weight_value = split_weight_by_tp_rank(
                        weight_value, split_axis=0, tp_rank_id=tp_rank_id,
                        tp_group_size=tp_group_size)
            else:
                weight_value = split_weight_by_tp_rank(
                    weight_value, split_axis=split_axis, tp_rank_id=tp_rank_id,
                    tp_group_size=tp_group_size)

            results[key] = weight_value
    return np.concatenate([results[key] for key in weight_mapping], axis=0)


def handle_training_qkv_weight(qkv_weight_value, config):
    """Processes the QKV (Query, Key, Value) weight matrix for attention layers.

    Args:
        qkv_weight_value (Tensor): The combined weight.
        config (dict or Config): Model configuration.

    Returns:
        A processed single tensor.
    """
    tp_group_size = get_tp_world_size()
    global_rank_id = get_rank()
    qkv_dim = len(qkv_weight_value.shape)
    w = qkv_weight_value.shape[0]
    if qkv_dim == 1:
        # cur qkv_weight is bias
        qkv_weight_value = qkv_weight_value.reshape(w, -1)
    head_dim = config.kv_channels if config.kv_channels else config.hidden_size // config.num_attention_heads
    q_channel = config.num_attention_heads * head_dim
    kv_channel = config.num_query_groups * head_dim

    q_weight = qkv_weight_value[:q_channel, :]
    k_weight = qkv_weight_value[q_channel:q_channel + kv_channel, :]
    v_weight = qkv_weight_value[q_channel + kv_channel:q_channel + 2 * kv_channel, :]
    q_weight = split_weight_by_tp_rank(q_weight, 0, global_rank_id, tp_group_size)
    # tp_size > kv_heads, the shape of kv weight will be replicated
    if tp_group_size > config.num_query_groups:
        replicate = tp_group_size // config.num_query_groups
        k_weight = split_weight_by_tp_rank(
            k_weight,
            split_axis=0,
            tp_rank_id=global_rank_id // replicate,
            tp_group_size=config.num_query_groups
        )
        v_weight = split_weight_by_tp_rank(
            v_weight,
            split_axis=0,
            tp_rank_id=global_rank_id // replicate,
            tp_group_size=config.num_query_groups
        )
    else:
        k_weight = split_weight_by_tp_rank(k_weight, 0, tp_rank_id=global_rank_id, tp_group_size=tp_group_size)
        v_weight = split_weight_by_tp_rank(v_weight, 0, tp_rank_id=global_rank_id, tp_group_size=tp_group_size)
    cat_qkv_weight = np.concatenate((q_weight, k_weight, v_weight), axis=0)
    if qkv_dim == 1:
        cat_qkv_weight = cat_qkv_weight.reshape(w // tp_group_size,)
    return cat_qkv_weight


def handle_training_ffn_weight(ffn_value):
    """Processes the FFN weight matrix for attention layers.

    Args:
        ffn_value (Tensor): The combined weight.

    Returns:
        A processed single tensor.
    """
    tp_group_size = get_tp_world_size()
    global_rank_id = get_rank()
    ffn_dim = len(ffn_value.shape)
    w = ffn_value.shape[0]
    if ffn_dim == 1:
        ffn_value = ffn_value.reshape(w, -1)
    w1_weight = ffn_value[: w // 2, :]
    w3_weight = ffn_value[w // 2: w // 2 * 2, :]
    w1_weight = split_weight_by_tp_rank(w1_weight, split_axis=0, tp_rank_id=global_rank_id, tp_group_size=tp_group_size)
    w3_weight = split_weight_by_tp_rank(w3_weight, split_axis=0, tp_rank_id=global_rank_id, tp_group_size=tp_group_size)
    cat_ffn_weight = np.concatenate((w1_weight, w3_weight), axis=0)
    if ffn_dim == 1:
        cat_ffn_weight = cat_ffn_weight.reshape(w // tp_group_size,)
    return cat_ffn_weight
