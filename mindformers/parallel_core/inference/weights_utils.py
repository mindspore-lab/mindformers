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
from safetensors import safe_open
import numpy as np

import mindspore as ms
from mindspore.communication.management import get_rank

from mindformers.parallel_core.inference.utils import get_tp_world_size


class WeightUtils:

    """A utility class for handling model weights (loading, converting, processing)."""

    def __init__(self):
        self.file_handles = {}
        self.tp_group_size = get_tp_world_size()
        self.global_rank_id = get_rank()
        self.mf_hf_mapping = {}
        self.mapping_dict = {}
        self.parameter_dict = {}
        self.processed_weights_keys = set()

    def get_file_handles(self, filename):
        if filename not in self.file_handles:
            fp = safe_open(filename, framework="np")
            self.file_handles[filename] = fp
        return self.file_handles[filename]

    def split_weight_by_tp_rank(self, weight, split_axis):
        """
        Split model weight by the current rank id for tensor parallelism.

        Args:
            weight(np.array): The full weight tensor to be split.
            split_axis(int): The axis along which to split the weight. Usually 0 (row-wise) or 1 (column-wise).

        Returns:
            weight: The split sub-tensor assigned to the current rank.
        """
        if isinstance(weight, np.ndarray):
            shape = weight.shape
        else:
            shape = weight.get_shape()
        split_size = shape[split_axis] // self.tp_group_size
        start = self.global_rank_id * split_size
        stop = (self.global_rank_id + 1) * split_size
        return weight[start:stop] if split_axis == 0 else weight[:, start:stop]

    def split_weight_specific_rank_with_resize(self, weight, split_axis, tp_rank_id, group_size):
        r"""
        Split model weight by a specified rank id with replication,
        used when tensor parallel size is greater than the number of kv heads.

        Args:
            weight(np.array): The full weight tensor to be split.
            split_axis(int): The axis along which to split the weight. Usually 0 (row-wise) or 1 (column-wise).
            tp_rank_id(int): The id of the current device/rank to which this split will be assigned.
            group_size(int): The number of query groups.

        Returns:
            weight: The split sub-tensor assigned to the current rank.
        """
        if isinstance(weight, np.ndarray):
            shape = weight.shape
        else:
            shape = weight.get_shape()
        split_size = shape[split_axis] // group_size
        start = tp_rank_id * split_size
        stop = (tp_rank_id + 1) * split_size
        return weight[start:stop] if split_axis == 0 else weight[:, start:stop]

    def deal_qkv(self, qkv_weight_value, config):
        """Processes the QKV (Query, Key, Value) weight matrix for attention layers.

        Args:
            qkv_weight_value (Tensor): The combined weight.
            config (dict or Config): Model configuration.

        Returns:
            A processed single tensor.
        """
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
        q_weight = self.split_weight_by_tp_rank(q_weight, 0)
        # tp_size > kv_heads, the shape of kv weight will be replicated
        if self.tp_group_size > config.num_query_groups:
            replicate = self.tp_group_size // config.num_query_groups
            k_weight = self.split_weight_specific_rank_with_resize(
                k_weight,
                split_axis=0,
                tp_rank_id=self.global_rank_id // replicate,
                group_size=config.num_query_groups
            )
            v_weight = self.split_weight_specific_rank_with_resize(
                v_weight,
                split_axis=0,
                tp_rank_id=self.global_rank_id // replicate,
                group_size=config.num_query_groups
            )
        else:
            k_weight = self.split_weight_by_tp_rank(k_weight, 0)
            v_weight = self.split_weight_by_tp_rank(v_weight, 0)
        cat_qkv_weight = np.concatenate((q_weight, k_weight, v_weight), axis=0)
        if qkv_dim == 1:
            cat_qkv_weight = cat_qkv_weight.reshape(w // self.tp_group_size,)
        return cat_qkv_weight

    def deal_ffn(self, ffn_value):
        """Processes the FFN weight matrix for attention layers.

        Args:
            ffn_value (Tensor): The combined weight.

        Returns:
            A processed single tensor.
        """
        ffn_dim = len(ffn_value.shape)
        w = ffn_value.shape[0]
        if ffn_dim == 1:
            ffn_value = ffn_value.reshape(w, -1)
        w1_weight = ffn_value[: w // 2, :]
        w3_weight = ffn_value[w // 2: w // 2 * 2, :]
        w1_weight = self.split_weight_by_tp_rank(w1_weight, 0)
        w3_weight = self.split_weight_by_tp_rank(w3_weight, 0)
        cat_ffn_weight = np.concatenate((w1_weight, w3_weight), axis=0)
        if ffn_dim == 1:
            cat_ffn_weight = cat_ffn_weight.reshape(w // self.tp_group_size,)
        return cat_ffn_weight

    def infer_trans_rope_weight(self, weight, qk_pos_emb_head_dim):
        """process rope router weight"""
        w1 = weight[..., -qk_pos_emb_head_dim::2, :]
        w2 = weight[..., -qk_pos_emb_head_dim + 1::2, :]
        weight[..., -qk_pos_emb_head_dim:, :] = np.concatenate([w1, w2], axis=-2)
        return weight

    def not_split(self, src_keys_dict, net_name, weights_path, config):
        for weight_name, file in src_keys_dict.items():
            weight_value = self.get_file_handles(f'{weights_path}/{file}').get_tensor(weight_name)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(weight_value).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)

    def split_by_tp_rank_rows(self, src_keys_dict, net_name, weights_path, config):
        for weight_name, file in src_keys_dict.items():
            weight_value = self.get_file_handles(f'{weights_path}/{file}').get_slice(weight_name)
            split_data = self.split_weight_by_tp_rank(weight_value, 1)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(split_data).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)

    def split_by_tp_rank_columns(self, src_keys_dict, net_name, weights_path, config):
        for weight_name, file in src_keys_dict.items():
            weight_value = self.get_file_handles(f'{weights_path}/{file}').get_slice(weight_name)
            split_data = self.split_weight_by_tp_rank(weight_value, 0)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(split_data).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)

    def split_qkv_weight(self, src_keys_dict, net_name, weights_path, config):
        """Splits and loads the combined QKV (Query, Key, Value) weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            weights_path: Path to the source checkpoint file.
            config: Model configuration.

        """
        if len(src_keys_dict) > 1:
            if len(src_keys_dict) != 3:
                raise ValueError(f'There should be three key values for q, k, v weights, '
                                 f'but {len(src_keys_dict)} are present')
            q_weight_value, k_weight_value, v_weight_value = None, None, None
            for weight_name, file in src_keys_dict.items():
                if weight_name.split('.')[-2] == self.mf_hf_mapping['linear_q']:
                    q_weight_value = self.get_file_handles(f'{weights_path}/{file}').get_slice(weight_name)
                    q_weight_value = self.split_weight_by_tp_rank(q_weight_value, 0)
                if weight_name.split('.')[-2] == self.mf_hf_mapping['linear_k']:
                    k_weight_value = self.get_file_handles(f'{weights_path}/{file}').get_slice(weight_name)
                if weight_name.split('.')[-2] == self.mf_hf_mapping['linear_v']:
                    v_weight_value = self.get_file_handles(f'{weights_path}/{file}').get_slice(weight_name)
            if q_weight_value is not None and k_weight_value is not None and v_weight_value is not None:
                if self.tp_group_size > config.num_query_groups:
                    replicate = self.tp_group_size // config.num_query_groups
                    k_weight_value = self.split_weight_specific_rank_with_resize(
                        k_weight_value, split_axis=0, tp_rank_id=self.global_rank_id // replicate,
                        group_size=config.num_query_groups
                    )
                    v_weight_value = self.split_weight_specific_rank_with_resize(
                        v_weight_value, split_axis=0, tp_rank_id=self.global_rank_id // replicate,
                        group_size=config.num_query_groups
                    )
                else:
                    k_weight_value = self.split_weight_by_tp_rank(k_weight_value, 0)
                    v_weight_value = self.split_weight_by_tp_rank(v_weight_value, 0)
                qkv_value = np.concatenate((q_weight_value, k_weight_value, v_weight_value), 0)
                self.parameter_dict[net_name] = ms.Parameter(
                    ms.from_numpy(qkv_value).astype(getattr(config, 'params_dtype')),
                    name=net_name, requires_grad=False)
            else:
                raise ValueError('The weight files are missing proper q/k/v weight matrices.')
        else:
            for weight_name, file in src_keys_dict.items():
                weight_value = self.get_file_handles(f'{weights_path}/{file}').get_tensor(weight_name)
                qkv_value = self.deal_qkv(weight_value, config)
                self.parameter_dict[net_name] = ms.Parameter(
                    ms.from_numpy(qkv_value).astype(getattr(config, 'params_dtype')),
                    name=net_name, requires_grad=False)

    def split_ffn_weight(self, src_keys_dict, net_name, weights_path, config):
        """Splits and loads the combined QKV (Query, Key, Value) weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            weights_path: Path to the source checkpoint file.
            config: Model configuration.

        """
        if len(src_keys_dict) > 1:
            if len(src_keys_dict) != 2:
                raise ValueError(f'There should be two key values for mlp weights, '
                                 f'but {len(src_keys_dict)} are present')
            ffn_weight_values = []
            for weight_name, file in src_keys_dict.items():
                weight_value = self.get_file_handles(f'{weights_path}/{file}').get_slice(weight_name)
                weight_value = self.split_weight_by_tp_rank(weight_value, 0)
                ffn_weight_values.append(weight_value)
            ffn_value = np.concatenate(ffn_weight_values, 0)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(ffn_value).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)
        else:
            for weight_name, file in src_keys_dict.items():
                weight_value = self.get_file_handles(f'{weights_path}/{file}').get_tensor(weight_name)
                ffn_value = self.deal_ffn(weight_value)
                self.parameter_dict[net_name] = ms.Parameter(
                    ms.from_numpy(ffn_value).astype(getattr(config, 'params_dtype')),
                    name=net_name, requires_grad=False)

    def split_router_expert_weight1(self, src_keys_dict, net_name, weights_path, config):
        """Splits and loads the combined QKV (Query, Key, Value) weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            weights_path: Path to the source checkpoint file.
            config: Model configuration.

        """
        w1_list = []
        w3_list = []
        sorted_items = sorted(
            src_keys_dict.items(),
            key=lambda x: int(x[0].split('.')[-3])
        )
        src_keys_dict = {}
        for k, v in sorted_items:
            src_keys_dict[k] = v
        for weight_name, file in src_keys_dict.items():
            weight_value = self.get_file_handles(f'{weights_path}/{file}').get_slice(weight_name)
            weight_value = self.split_weight_by_tp_rank(weight_value, 0)
            if self.mf_hf_mapping['gating'] in weight_name:
                w1_list.append(weight_value)
            else:
                w3_list.append(weight_value)

        w1_value_all = np.stack(w1_list, axis=0)
        w3_value_all = np.stack(w3_list, axis=0)
        linear_fc1_value = np.concatenate([w1_value_all, w3_value_all], axis=1)
        linear_fc1_value = ms.from_numpy(linear_fc1_value).permute(
            0, 2, 1).astype(dtype=getattr(config, 'params_dtype'))
        self.parameter_dict[net_name] = ms.Parameter(linear_fc1_value, name=net_name, requires_grad=False)

    def split_router_expert_weight2(self, src_keys_dict, net_name, weights_path, config):
        """Splits and loads the combined QKV (Query, Key, Value) weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            weights_path: Path to the source checkpoint file.
            config: Model configuration.

        """
        w2_list = []
        sorted_items = sorted(
            src_keys_dict.items(),
            key=lambda x: int(x[0].split('.')[-3])
        )
        src_keys_dict = {}
        for k, v in sorted_items:
            src_keys_dict[k] = v
        for weight_name, file in src_keys_dict.items():
            weight_value = self.get_file_handles(f'{weights_path}/{file}').get_slice(weight_name)
            weight_value = self.split_weight_by_tp_rank(weight_value, 1)
            w2_list.append(weight_value)

        w2_value_all = np.stack(w2_list, axis=0)
        linear_fc2_value = ms.from_numpy(w2_value_all).permute(
            0, 2, 1).astype(dtype=getattr(config, 'params_dtype'))
        self.parameter_dict[net_name] = ms.Parameter(linear_fc2_value, name=net_name, requires_grad=False)

    def split_linear_qkv_down_proj(self, src_keys_dict, net_name, weights_path, config):
        """Splits and loads the combined QKV (Query, Key, Value) weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            weights_path: Path to the source checkpoint file.
            config: Model configuration.

        """
        q_down_value, kv_down_value = None, None
        for weight_name, file in src_keys_dict.items():
            if weight_name.split('.')[-2] == self.mf_hf_mapping['linear_q_down_proj']:
                q_down_value = self.get_file_handles(f'{weights_path}/{file}').get_tensor(weight_name)
            if weight_name.split('.')[-2] == self.mf_hf_mapping['linear_kv_down_proj']:
                kv_down_value = self.get_file_handles(f'{weights_path}/{file}').get_tensor(weight_name)
                kv_lora_rank = config.kv_lora_rank
                qk_rope_head_dim = config.qk_pos_emb_head_dim
                kv_head_dim = kv_lora_rank + qk_rope_head_dim
                kv_down_value = kv_down_value.reshape(kv_head_dim, -1)
                kv_down_value = self.infer_trans_rope_weight(kv_down_value, qk_rope_head_dim)
        if q_down_value is not None and kv_down_value is not None:
            linear_qkv_down_proj_value = np.concatenate((q_down_value, kv_down_value), 0)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(linear_qkv_down_proj_value).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)
        else:
            ValueError('The weight files are missing proper linear_q_down_proj/linear_kv_down_proj weight matrices.')

    def split_linear_kv_down_proj(self, src_keys_dict, net_name, weights_path, config):
        """Splits and loads the combined QKV (Query, Key, Value) weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            weights_path: Path to the source checkpoint file.
            config: Model configuration.

        """
        for weight_name, file in src_keys_dict.items():
            weight_value = self.get_file_handles(f'{weights_path}/{file}').get_tensor(weight_name)
            kv_lora_rank = config.kv_lora_rank
            qk_rope_head_dim = config.qk_pos_emb_head_dim
            kv_head_dim = kv_lora_rank + qk_rope_head_dim
            weight_value = weight_value.reshape(kv_head_dim, -1)
            weight_value = self.infer_trans_rope_weight(weight_value, qk_rope_head_dim)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(weight_value).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)

    def split_linear_q_up_proj(self, src_keys_dict, net_name, weights_path, config):
        """Splits and loads the combined QKV (Query, Key, Value) weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            weights_path: Path to the source checkpoint file.
            config: Model configuration.

        """
        for weight_name, file in src_keys_dict.items():
            linear_q_up_proj_value = self.get_file_handles(f'{weights_path}/{file}').get_tensor(weight_name)
            num_heads = config.num_attention_heads
            rope_dim = config.qk_pos_emb_head_dim + config.qk_head_dim
            linear_q_up_proj_value = linear_q_up_proj_value.reshape(num_heads, rope_dim, -1)
            linear_q_up_proj_value = self.infer_trans_rope_weight(linear_q_up_proj_value, config.qk_pos_emb_head_dim)
            linear_q_up_proj_value = linear_q_up_proj_value.reshape(num_heads * rope_dim, -1)
            linear_q_up_proj_value = self.split_weight_by_tp_rank(linear_q_up_proj_value, split_axis=0)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(linear_q_up_proj_value).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)

    def split_linear_kv_up_proj(self, src_keys_dict, net_name, weights_path, config):
        """Splits and loads the combined QKV (Query, Key, Value) weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            weights_path: Path to the source checkpoint file.
            config: Model configuration.

        """
        for weight_name, file in src_keys_dict.items():
            linear_kv_up_proj_value = self.get_file_handles(f'{weights_path}/{file}').get_tensor(weight_name)
            qk_nope_head_dim = config.qk_head_dim
            v_head_dim = config.v_head_dim
            lkv2kv_head = qk_nope_head_dim + v_head_dim
            num_heads = config.num_attention_heads
            linear_kv_up_proj_value = linear_kv_up_proj_value.reshape(num_heads, lkv2kv_head, -1)
            value_k_nope, value_v = (linear_kv_up_proj_value[:, :qk_nope_head_dim, :],
                                     linear_kv_up_proj_value[:, qk_nope_head_dim:, :])
            # value_k_nope
            value_k_nope = value_k_nope.reshape(-1, value_k_nope.shape[-1])
            value_k_nope = self.split_weight_by_tp_rank(value_k_nope, split_axis=0)
            # value_v_nope
            value_v = value_v.reshape(-1, value_v.shape[-1])
            value_v = self.split_weight_by_tp_rank(value_v, split_axis=0)
            linear_kv_up_proj_value = np.concatenate((value_k_nope, value_v), 0)
            self.parameter_dict[net_name] = ms.Parameter(
                ms.from_numpy(linear_kv_up_proj_value).astype(getattr(config, 'params_dtype')),
                name=net_name, requires_grad=False)

    def split_shared_experts(self, src_keys_dict, net_name, weights_path, config):
        """Splits and loads the combined QKV (Query, Key, Value) weights from source checkpoint.

        Args:
            src_keys_dict: Dictionary mapping source weight names to their values in the checkpoint.
            net_name: Name of the target network module where weights will be loaded.
            weights_path: Path to the source checkpoint file.
            config: Model configuration.

        """
        weight_values = []
        for weight_name, file in src_keys_dict.items():
            weight_value = self.get_file_handles(f'{weights_path}/{file}').get_slice(weight_name)
            weight_value = self.split_weight_by_tp_rank(weight_value, split_axis=0)
            weight_values.append(weight_value)
        np_data = np.concatenate(weight_values, 0)
        self.parameter_dict[net_name] = ms.Parameter(
            ms.from_numpy(np_data).astype(getattr(config, 'params_dtype')), name=net_name, requires_grad=False)
