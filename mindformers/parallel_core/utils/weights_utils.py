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
from mindspore.communication.management import get_rank, get_group_size
from mindformers.parallel_core.inference.transformer.mlp import MLP
from mindformers.parallel_core.inference.transformer.attention import SelfAttention
from mindformers.parallel_core.inference.transformer.multi_latent_attention import MLASelfAttention
from mindformers.parallel_core.inference.transformer.moe.moe_layer import MoELayer
from mindformers.parallel_core.inference.transformer.norm import RMSNorm, LayerNorm
from mindformers.parallel_core.inference.transformer.identity_op import IdentityOp


def add_param(parameter_dict, config, name, weights_path, value, mf_hf_map, split_axis=None):
    r"""
    Load weights other than transformer layer.

    Args:
        parameter_dict: Weight dict loaded into the network.
        config: Config of TransFormer Config.
        name: Name of weight.
        weights_path: Weight path.
        value: Value of weight.
        mf_hf_map: Mapping table of hf weight and mf weight.
        split_axis: Whether weight is concated.

    """
    np_data = get_safetensor_from_file(
        config, name, weights_path, value, mf_hf_map,
        is_split_param=split_axis is not None, split_axis=split_axis
    )
    parameter_dict[name] = ms.Parameter(
        ms.Tensor(np_data, getattr(config, 'params_dtype')), name=name, requires_grad=False
    )


def deal_non_layers_weights(parameter_dict, config, weights_path, weights, mf_hf_map):
    r"""
    Load weights other than transformer layer.

    Args:
        parameter_dict: Weight dict loaded into the network.
        config: Config of TransFormer Config.
        weights_path: Weight path.
        weights: Dict of weight.
        mf_hf_map: Mapping table of hf weight and mf weight.

    """
    weight_name = f"output_layer.weight"
    add_param(parameter_dict, config, weight_name, weights_path, weights[weight_name], mf_hf_map, 0)
    weight_name = f"embedding.word_embeddings.weight"
    add_param(parameter_dict, config, weight_name, weights_path, weights[weight_name], mf_hf_map, 0)
    weight_name = "decoder.final_norm.weight"
    add_param(parameter_dict, config, weight_name, weights_path, weights[weight_name], mf_hf_map)


def deal_attention_weights(layer_obj, parameter_dict, config, weights_path, weights, mf_hf_map, layer_id,
                           source_qkv_concat):
    r"""
    Load weights of attention module.

    Args:
        layer_obj: SelfAttention object in TransFormer Layer.
        parameter_dict: Weight dict loaded into the network.
        config: Config of TransFormer Config.
        weights_path: Weight path.
        weights: Dict of weight.
        mf_hf_map: Mapping table of hf weight and mf weight.
        layer_id: Current layers.
        source_qkv_concat: Whether qkv is concated.

    """
    if isinstance(layer_obj.self_attention, SelfAttention):
        q_norm_name = f"decoder.layers.{layer_id}.self_attention.q_layernorm.weight"
        add_param(parameter_dict, config, q_norm_name, weights_path, weights[q_norm_name], mf_hf_map)
        k_norm_name = f"decoder.layers.{layer_id}.self_attention.k_layernorm.weight"
        add_param(parameter_dict, config, k_norm_name, weights_path, weights[k_norm_name], mf_hf_map)
        linear_proj_name = f"decoder.layers.{layer_id}.self_attention.linear_proj.weight"
        add_param(parameter_dict, config, linear_proj_name, weights_path, weights[linear_proj_name], mf_hf_map, 1)
        if not source_qkv_concat:
            wq_key = f"decoder.layers.{layer_id}.self_attention.linear_q.weight"
            w_qkv_key, w_qkv_value = concat_qkv_weight(wq_key, weights_path, weights, mf_hf_map)
            add_param(parameter_dict, config, w_qkv_key, weights_path, w_qkv_value, mf_hf_map, 0)
        else:
            linear_qkv_name = f"decoder.layers.{layer_id}.self_attention.linear_qkv.weight"
            add_param(parameter_dict, config, linear_qkv_name, weights_path, weights[linear_qkv_name], mf_hf_map, 0)


def deal_mlp_weights(layer_obj, parameter_dict, config, weights_path, weights, mf_hf_map, layer_id,
                     source_qkv_concat):
    r"""
    Load weights of mlp module.

    Args:
        layer_obj: MLP object in TransFormer Layer.
        parameter_dict: Weight dict loaded into the network.
        config: Config of TransFormer Config.
        weights_path: Weight path.
        weights: Dict of weight.
        mf_hf_map: Mapping table of hf weight and mf weight.
        layer_id: Current layers.
        source_qkv_concat: Whether qkv is concated.

    """
    if isinstance(layer_obj.mlp, MLP):
        linear_fc2_name = f"decoder.layers.{layer_id}.mlp.linear_fc2.weight"
        add_param(parameter_dict, config, linear_fc2_name, weights_path, weights[linear_fc2_name], mf_hf_map, 1)
        if not source_qkv_concat:
            w1_key = f"decoder.layers.{layer_id}.mlp.gating.weight"
            w_gate_hidden_key, w_gate_hidden_value = concat_ffn_weight(w1_key, weights_path, weights, mf_hf_map)
            add_param(parameter_dict, config, w_gate_hidden_key, weights_path, w_gate_hidden_value, mf_hf_map, 0)
        else:
            linear_fc1_name = f"decoder.layers.{layer_id}.mlp.linear_fc1.weight"
            add_param(parameter_dict, config, linear_fc1_name, weights_path, weights[linear_fc1_name], mf_hf_map, 0)


def deal_moe_weights(layer_obj, parameter_dict, config, weights_path, weights, mf_hf_map, layer_id):
    r"""
    Load weights of moe module.

    Args:
        layer_obj: MOE object in TransFormer Layer.
        parameter_dict: Weight dict loaded into the network.
        config: Config of TransFormer Config.
        weights_path: Weight path.
        weights: Dict of weight.
        mf_hf_map: Mapping table of hf weight and mf weight.
        layer_id: Current layers.

    """
    if isinstance(layer_obj.mlp, MoELayer):
        num_router_experts = config.num_moe_experts
        num_shared_experts = config.shared_expert_num

        if config.moe_router_enable_expert_bias:
            mlp_bias_name = f"decoder.layers.{layer_id}.mlp.router.expert_bias"
            add_param(parameter_dict, config, mlp_bias_name, weights_path, weights[mlp_bias_name], mf_hf_map)

        router_dense_hf_name = f"decoder.layers.{layer_id}.mlp.router.weight.weight"
        add_param(parameter_dict, config, router_dense_hf_name, weights_path, weights[router_dense_hf_name],
                  mf_hf_map)

        # router_experts
        w1_list = []
        w2_list = []
        w3_list = []

        for index in range(0, num_router_experts):
            w1_key = f"decoder.layers.{layer_id}.mlp.experts.{index}.gating.weight"
            w1_value = get_safetensor_from_file(config, w1_key, weights_path, weights[w1_key], mf_hf_map,
                                                is_split_param=True, split_axis=0)
            w2_key = f"decoder.layers.{layer_id}.mlp.experts.{index}.linear_fc2.weight"
            w2_value = get_safetensor_from_file(config, w2_key, weights_path, weights[w2_key], mf_hf_map,
                                                is_split_param=True, split_axis=1)
            w3_key = f"decoder.layers.{layer_id}.mlp.experts.{index}.linear_fc1.weight"
            w3_value = get_safetensor_from_file(config, w3_key, weights_path, weights[w3_key], mf_hf_map,
                                                is_split_param=True, split_axis=0)

            w1_list.append(w1_value)
            w2_list.append(w2_value)
            w3_list.append(w3_value)

        w1_value_all = np.stack(w1_list, axis=0)
        w2_value_all = np.stack(w2_list, axis=0)
        w3_value_all = np.stack(w3_list, axis=0)
        linear_fc1_name = f"decoder.layers.{layer_id}.mlp.experts.weight1"
        linear_fc1_value = np.concatenate([w1_value_all, w3_value_all], axis=1)
        linear_fc1_value = ms.from_numpy(linear_fc1_value).permute(
            0, 2, 1).astype(dtype=ms.bfloat16)
        parameter_dict[linear_fc1_name] = ms.Parameter(
            ms.Tensor(linear_fc1_value, getattr(config, 'params_dtype')), name=linear_fc1_name, requires_grad=False
        )

        linear_fc2_name = f"decoder.layers.{layer_id}.mlp.experts.weight2"
        linear_fc2_value = ms.from_numpy(w2_value_all).permute(
            0, 2, 1).astype(dtype=ms.bfloat16)
        parameter_dict[linear_fc2_name] = ms.Parameter(
            ms.Tensor(linear_fc2_value, getattr(config, 'params_dtype')), name=linear_fc2_name, requires_grad=False
        )

        # shared_experts
        if num_shared_experts > 0:
            w1_key = f"decoder.layers.{layer_id}.mlp.shared_experts.gating.weight"
            w3_key, w3_value = concat_ffn_weight(w1_key, weights_path, weights, mf_hf_map)
            add_param(parameter_dict, config, w3_key, weights_path, w3_value, mf_hf_map, 0)
            w2_key = f"decoder.layers.{layer_id}.mlp.shared_experts.linear_fc2.weight"
            add_param(parameter_dict, config, w2_key, weights_path, weights[w2_key], mf_hf_map, 1)


def deal_mla_weights(layer_obj, parameter_dict, config, weights_path, weights, mf_hf_map, layer_id):
    r"""
    Load weights of mla module.

    Args:
        layer_obj: MLASelfAttention object in TransFormer Layer.
        parameter_dict: Weight dict loaded into the network.
        config: Config of TransFormer Config.
        weights_path: Weight path.
        weights: Dict of weight.
        mf_hf_map: Mapping table of hf weight and mf weight.
        layer_id: Current layers.

    """
    if isinstance(layer_obj.self_attention, MLASelfAttention):
        linear_q_down_proj_name = f"decoder.layers.{layer_id}.self_attention.linear_q_down_proj.weight"
        linear_q_layernorm = f"decoder.layers.{layer_id}.self_attention.q_layernorm.weight"
        linear_kv_layernorm = f"decoder.layers.{layer_id}.self_attention.kv_layernorm.weight"
        linear_proj_name = f"decoder.layers.{layer_id}.self_attention.linear_proj.weight"
        linear_q_up_proj_name = f"decoder.layers.{layer_id}.self_attention.linear_q_up_proj.weight"
        linear_kv_down_proj_name = f"decoder.layers.{layer_id}.self_attention.linear_kv_down_proj.weight"
        linear_kv_up_proj_name = f"decoder.layers.{layer_id}.self_attention.linear_kv_up_proj.weight"

        linear_qkv_down_proj_name = f"decoder.layers.{layer_id}.self_attention.linear_qkv_down_proj.weight"

        with safe_open(f"{weights_path}/{weights[linear_q_down_proj_name]}", framework="np") as sf_file:
            linear_q_down_proj_value = sf_file.get_tensor(mf_hf_map.get(linear_q_down_proj_name))
        with safe_open(f"{weights_path}/{weights[linear_kv_down_proj_name]}", framework="np") as sf_file:
            linear_kv_down_proj_value = sf_file.get_tensor(mf_hf_map.get(linear_kv_down_proj_name))
        if config.q_lora_rank is not None:
            linear_qkv_down_proj_value = np.concatenate(
                (linear_q_down_proj_value, linear_kv_down_proj_value), 0)
            add_param(parameter_dict, config, linear_qkv_down_proj_name, weights_path, linear_qkv_down_proj_value,
                      mf_hf_map, 0)
        else:
            linear_q_proj_name = f"decoder.layers.{layer_id}.self_attention.linear_q_proj.weight"
            add_param(parameter_dict, config, linear_q_proj_name, weights_path, linear_q_down_proj_value,
                      mf_hf_map, 0)
            add_param(parameter_dict, config, linear_kv_down_proj_name, weights_path, linear_kv_down_proj_value,
                      mf_hf_map, 0)
        add_param(parameter_dict, config, linear_q_layernorm, weights_path, weights[linear_q_layernorm], mf_hf_map)

        add_param(parameter_dict, config, linear_kv_layernorm, weights_path,
                  weights[linear_kv_layernorm], mf_hf_map)

        add_param(parameter_dict, config, linear_proj_name, weights_path, weights[linear_proj_name],
                  mf_hf_map, 1)

        add_param(parameter_dict, config, linear_q_up_proj_name, weights_path, weights[linear_q_up_proj_name],
                  mf_hf_map, 0)

        add_param(parameter_dict, config, linear_kv_up_proj_name, weights_path, weights[linear_kv_up_proj_name],
                  mf_hf_map, 0)


def deal_layer_norm_weights(layer_obj, parameter_dict, config, weights_path, weights, mf_hf_map, layer_id):
    r"""
    Load weights of RMSNorm module.

    Args:
        layer_obj: RMSNorm object or LayerNorm object in TransFormer Layer.
        parameter_dict: Weight dict loaded into the network.
        config: Config of TransFormer Config.
        weights_path: Weight path.
        weights: Dict of weight.
        mf_hf_map: Mapping table of hf weight and mf weight.
        layer_id: Current layers.

    """
    if not isinstance(layer_obj.input_layernorm, IdentityOp):
        if isinstance(layer_obj.input_layernorm, RMSNorm):
            input_layernorm_name = f"decoder.layers.{layer_id}.input_layernorm.weight"
            add_param(parameter_dict, config, input_layernorm_name, weights_path, weights[input_layernorm_name],
                      mf_hf_map)
        elif isinstance(layer_obj.input_layernorm, LayerNorm):
            raise NotImplementedError("LayerNorm is not implemented")
    if not isinstance(layer_obj.pre_mlp_layernorm, IdentityOp):
        if isinstance(layer_obj.pre_mlp_layernorm, RMSNorm):
            pre_mlp_layernorm = f"decoder.layers.{layer_id}.pre_mlp_layernorm.weight"
            add_param(parameter_dict, config, pre_mlp_layernorm, weights_path, weights[pre_mlp_layernorm], mf_hf_map)
        elif isinstance(layer_obj.pre_mlp_layernorm, LayerNorm):
            raise NotImplementedError("LayerNorm is not implemented")


def concat_qkv_weight(wq_key, weights_path, weights, mf_hf_map):
    r"""
    concat qkv weight from dicts.

    Args:
        wq_key: query weight name.
        weights_path: the path of storing weights.
        weights: weight dict.
        mf_hf_map: the key is the value of the mf weights and the value is the value of the hf weights.

    Returns:

    """
    wk_key = wq_key.replace('linear_q', 'linear_k')
    wv_key = wq_key.replace('linear_q', 'linear_v')
    wq_value = weights.pop(wq_key)
    wk_value = weights.pop(wk_key, None)
    wv_value = weights.pop(wv_key, None)
    if isinstance(wq_value, str) and wq_value.endswith('safetensors'):
        with safe_open(f"{weights_path}/{wq_value}", framework="np") as sf_file:
            wq_value = sf_file.get_tensor(mf_hf_map.get(wq_key))
        with safe_open(f"{weights_path}/{wk_value}", framework="np") as sf_file:
            wk_value = sf_file.get_tensor(mf_hf_map.get(wk_key))
        with safe_open(f"{weights_path}/{wv_value}", framework="np") as sf_file:
            wv_value = sf_file.get_tensor(mf_hf_map.get(wv_key))
    w_qkv_key = wq_key.replace('linear_q', 'linear_qkv')
    w_qkv_value = np.concatenate((wq_value, wk_value, wv_value), 0)
    return w_qkv_key, w_qkv_value


def concat_ffn_weight(w1_key, weights_path, weights, mf_hf_map):
    r"""
    concat ffn weight from dicts.

    Args:
        w1_key: ffn w1 weight name.
        weights_path: the path of storing weights.
        weights: weight dict.
        mf_hf_map: the key is the value of the mf weights and the value is the value of the hf weights.

    Returns:

    """
    w3_key = w1_key.replace('gating', 'linear_fc1')
    w1_value = weights.pop(w1_key)
    w3_value = weights.pop(w3_key, None)
    if isinstance(w1_value, str) and w1_value.endswith('safetensors'):
        with safe_open(f"{weights_path}/{w1_value}", framework="np") as sf_file:
            w1_value = sf_file.get_tensor(mf_hf_map.get(w1_key))
        with safe_open(f"{weights_path}/{w3_value}", framework="np") as sf_file:
            w3_value = sf_file.get_tensor(mf_hf_map.get(w3_key))
    w_gate_hidden_key = w1_key.replace('gating', 'linear_fc1')
    w_gate_hidden_value = np.concatenate((w1_value, w3_value), 0)
    return w_gate_hidden_key, w_gate_hidden_value


def get_safetensor_from_file(config, param_name, weights_path, weight, mf_hf_map, is_split_param=False,
                             split_axis=0):
    r"""
    The weight is processed in modules, and the weight is cut online and loaded.

    Args:
        param_name: The key of weights.
        weights_path: The path of storing weights.
        weight: A dict for storing keys and value of weights.
        mf_hf_map: A dict for storing keys of huggingface weight and Mindformers weight.
        is_split_param: Split or not.
        split_axis: According to the first dimension or the second dimension to split.

    Returns:
        np_data: Data after split.
    """
    tp_group_size = get_group_size()
    rank_id = get_rank()

    def split(tensor):
        split_size = tensor.shape[split_axis] // tp_group_size
        start = rank_id * split_size
        stop = (rank_id + 1) * split_size
        return tensor[start:stop] if split_axis == 0 else tensor[:, start:stop]

    def deal_qkv(np_data, config):
        qkv_dim = len(np_data.shape)
        w = np_data.shape[0]
        if qkv_dim == 1:
            # cur qkv_weight is bias
            np_data = np_data.reshape(w, -1)
        head_dim = config.kv_channels if config.kv_channels else config.hidden_size // config.num_attention_heads
        q_channel = config.num_attention_heads * head_dim
        kv_channel = config.num_query_groups * head_dim

        q_weight = np_data[:q_channel, :]
        k_weight = np_data[q_channel:q_channel + kv_channel, :]
        v_weight = np_data[q_channel + kv_channel:q_channel + 2 * kv_channel, :]
        q_weight = split(q_weight)
        k_weight = split(k_weight)
        v_weight = split(v_weight)
        cat_qkv_weight = np.concatenate((q_weight, k_weight, v_weight), axis=0)
        if qkv_dim == 1:
            cat_qkv_weight = cat_qkv_weight.reshape(w // tp_group_size,)
        return cat_qkv_weight

    def deal_ffn(np_data):
        ffn_dim = len(np_data.shape)
        w = np_data.shape[0]
        if ffn_dim == 1:
            np_data = np_data.reshape(w, -1)
        w1_weight = np_data[: w // 2, :]
        w3_weight = np_data[w // 2: w // 2 * 2, :]
        w1_weight = split(w1_weight)
        w3_weight = split(w3_weight)
        cat_ffn_weight = np.concatenate((w1_weight, w3_weight), axis=0)
        if ffn_dim == 1:
            cat_ffn_weight = cat_ffn_weight.reshape(w // tp_group_size,)
        return cat_ffn_weight

    if isinstance(weight, str) and weight.endswith('safetensors'):
        param_weight = mf_hf_map.get(param_name)
        with safe_open(f"{weights_path}/{weight}", framework="np") as sf_file:
            np_data = sf_file.get_tensor(param_weight)
            if not is_split_param:
                return np_data
            if param_name.split('.')[-2] == 'linear_qkv':
                return deal_qkv(np_data, config)
            if param_name.split('.')[-2] == 'linear_fc1' and param_name.split('.')[-3] == 'mlp':
                return deal_ffn(np_data)
            return split(np_data)
    if not is_split_param:
        np_data = weight
        return np_data
    np_data = weight
    np_dim = len(np_data.shape)
    if param_name.split('.')[-2] == 'linear_qkv':
        return deal_qkv(np_data, config)
    if param_name.split('.')[-2] == 'linear_fc1':
        return deal_ffn(np_data)
    return split(np_data).reshape(-1) if np_dim == 1 else split(np_data)
