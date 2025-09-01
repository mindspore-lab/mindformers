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
"""Utility functions for TransformerConfig."""

from dataclasses import asdict
from copy import deepcopy
from typing import Union
import types

from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.parallel_core.transformer_config import TransformerConfig, MLATransformerConfig
from mindformers.tools.logger import logger
from mindformers.tools.register.template import ParallelConfig


def is_float_32(dtype: Union[str, bool]) -> bool:
    """
    This method to check whether the dtype is float32.

    Args:
        dtype (Union[str, bool]): A string of dtype, of a bool flag of dtype.

    Returns:
        A bool flag.
        If the dtype satisfies the fp32 validation requirements, return `True`. Otherwise, return 'False'.
    """
    if isinstance(dtype, str):
        is_fp32 = dtype in ["fp32", "float32"]
    elif isinstance(dtype, bool):
        is_fp32 = dtype
    else:
        raise TypeError(f"The dtype should be a string in ['fp32', 'float32'], or a bool flag, "
                        f"bot got '{type(dtype)}'.")

    return is_fp32


def is_use_gated_sigmoid(score_func: [bool, str]) -> str:
    """
    This method to check whether to use gated 'sigmoid' or 'softmax'.

    Args:
        score_func (bool): A flag that indicate whether to use gated sigmoid.

    Returns:
        A string to choose the function for MoE router score.
    """
    if isinstance(score_func, str):
        return score_func
    if score_func:
        return "sigmoid"
    return "softmax"


def get_recompute(recompute):
    """Get the recompute configs."""
    if isinstance(recompute, bool):
        return recompute
    return recompute['recompute']


def get_cp_comm_type(context_parallel_algo: str):
    """
    This method to check types of 'context_parallel_algo'.

    Args:
        context_parallel_algo (str): Algorithm to use for context parallelism.
        Can be "colossalai_cp", "ulysses_cp" or "hybrid_cp". Only effective when context_parallel > 1

    Returns:
        A string to choose the function for context parallelism.
        cp_comm_type can be "p2p" or "all_gather" or "a2a" or "a2a+p2p".
    """
    context_parallelism_mapping = {
        "colossalai_cp": "all_gather",
        "ulysses_cp": "a2a",
    }

    if context_parallel_algo not in context_parallelism_mapping:
        raise ValueError(f"The context_parallel_algo {context_parallel_algo} is not supported, "
                         f"context_parallel_algo only support colossalai_cp, ulysses_cp or hybrid_cp")
    cp_comm_type = context_parallelism_mapping[context_parallel_algo]

    return cp_comm_type


def get_drop_policy(drop_policy: Union[bool, str]):
    if isinstance(drop_policy, bool):
        return 'position' if drop_policy else 'probs'
    if isinstance(drop_policy, str):
        return drop_policy
    raise TypeError(f"drop_policy (enable_sdrop, moe_token_drop_policy) should be bool or str, "
                    f"but get {type(drop_policy)}.")


def scatter_multi_mapping_keys_to_mapping(mapping):
    """
    Expand multiple mapping relationships contained in `convert_map`.

    Args:
        mapping (dict): A dict contains the mapping to convert the keys of `model_config`.

    Returns:
        A dict with expand of all mappings.
    """
    new_mapping = {}

    for k, v in mapping.items():
        if not isinstance(k, tuple):
            new_mapping[k] = v
            continue

        for multi_key in k:
            new_mapping[multi_key] = v

    return new_mapping


def update_addtional_map_to_mapping(mapping, additional_map):
    """
    Update the `additional_map` into the `convert_mapping`.

    Args:
        mapping (dict): A dict contains the mapping to convert the keys of `model_config`.
        additional_map (dict): A dict contains the additional mapping to convert the keys of `model_config`,
                                passed in by user.

    Returns:
        A dict contains the mapping to convert the keys of `model_config`.
    """
    if additional_map is None:
        return mapping

    for k, v in additional_map.items():
        if not isinstance(k, str):
            raise TypeError(f"Key in additional_map should be 'str', but get '{v}'.")

        if isinstance(v, str):
            pass
        elif isinstance(v, tuple):
            if len(v) != 2 or not isinstance(v[0], str) or not isinstance(v[1], types.FunctionType):
                raise TypeError(f"Value in additional_map should be 'str' or '(str, function)', but get '{v}'.")
        else:
            raise TypeError(f"Value in additional_map should be 'str' or '(str, function)', but get '{v}'.")
        # update mapping
        mapping[k] = v

    return mapping


def get_reversed_mapping(mapping):
    """
    Reverse the mapping table, to print information for multiple configurations at the same time.

    Args:
         mapping (dict): A dict contains the mapping to convert the keys of `model_config`.

    Returns:
        A dict contains the reversed convert mapping.
    """
    reversed_mapping = {}
    for k, v in mapping.items():
        if isinstance(v, tuple):
            v = v[0]
        if v not in reversed_mapping:
            reversed_mapping[v] = k
        elif isinstance(reversed_mapping[v], str):
            reversed_mapping[v] = (reversed_mapping[v], k)
        else:
            reversed_mapping[v] = reversed_mapping[v] + (k,)
    return reversed_mapping


DEFAULT_WHITE_KEY = set()
PRETRAIN_CONFIG_KEY = set(PretrainedConfig().to_dict().keys())
PARALLEL_CONFIG_KEY = set(ParallelConfig.keys())
INFER_CONFIG_KEY = set({
    "mindformers_version", "rl_config", "checkpoint_name_or_path", "_name_or_path", "type", "model_type",
    "tokenizer_class", "architectures", "is_encoder_decoder", "is_sample_acceleration", "bos_token_id", "eos_token_id",
    "temperature", "repetition_penalty", "max_decode_length", "top_k", "top_p", "do_sample"
})
DEFAULT_WHITE_KEY.update(PRETRAIN_CONFIG_KEY)
DEFAULT_WHITE_KEY.update(PARALLEL_CONFIG_KEY)
DEFAULT_WHITE_KEY.update(INFER_CONFIG_KEY)
DEFAULT_WHITE_KEY.update({
    'monitor_config', 'dataset_config', 'batch_size', 'multiple_of', 'ffn_dim_multiplier', 'qkv_concat', 'use_past',
    'scaling_factor', 'input_sliced_sig', 'return_extra_loss', 'moe_config'
})

COMMON_CONFIG_MAPPING = {
    #####################################################################################
    # Maps the configuration keys on the left to the TransformerConfig keys on the right.
    #
    # If the format on the right of mapping value is similar to '("Key", trans_func)',
    # then the corresponding value to be converted is obtained through `trans_func`.
    #####################################################################################

    # ModelParallelConfig
    # Model parallelism
    ("data_parallel", "data_parallel_size"): "data_parallel_size",
    ("model_parallel", "tensor_model_parallel_size"): "tensor_model_parallel_size",
    ("pipeline_stage", "pipeline_model_parallel_size"): "pipeline_model_parallel_size",
    ("pp_interleave_num", "virtual_pipeline_model_parallel_size"): "virtual_pipeline_model_parallel_size",
    ("use_seq_parallel", "sequence_parallel"): "sequence_parallel",
    ("context_parallel", "context_parallel_size"): "context_parallel_size",
    ("expert_parallel", "expert_model_parallel_size"): "expert_model_parallel_size",
    ("expert_model_parallel", "expert_tensor_parallel_size"): "expert_tensor_parallel_size",
    # not changes
    "micro_batch_num": "micro_batch_num",
    "seq_split_num": "seq_split_num",
    "gradient_aggregation_group": "gradient_aggregation_group",
    "offset": "offset",
    "vocab_emb_dp": "vocab_emb_dp",

    # Training
    ("param_init_type", "params_dtype"): "params_dtype",
    # not changes
    "finalize_model_grads_func": "finalize_model_grads_func",
    "grad_scale_func": "grad_scale_func",
    "grad_sync_func": "grad_sync_func",
    "param_sync_func": "param_sync_func",
    "num_microbatches_with_partial_activation_checkpoints": "num_microbatches_with_partial_activation_checkpoints",
    "print_separate_loss": "print_separate_loss",

    # CPU Offloading
    ("swap", "cpu_offloading"): "cpu_offloading",
    ("layer_swap", "cpu_offloading_num_layers"): "cpu_offloading_num_layers",
    # not changes
    "op_swap": "op_swap",
    "default_prefetch": "default_prefetch",

    # TransformerConfig
    # Model Architecture
    ("mtp_depth", "num_nextn_predict_layers", "mtp_num_layers"): "mtp_num_layers",
    ("mtp_loss_factor", "mtp_loss_scaling_factor"): "mtp_loss_scaling_factor",
    ("num_heads", "num_attention_heads", "n_head"): "num_attention_heads",
    ("n_kv_heads", "num_key_value_heads", "num_query_groups"): "num_query_groups",
    ("intermediate_size", "ffn_hidden_size"): "ffn_hidden_size",
    ("head_dim", "kv_channels"): "kv_channels",
    ("residual_dtype", "fp32_residual_connection"): (
        "fp32_residual_connection", is_float_32
    ),
    ("rms_norm_eps", "layernorm_epsilon", "layer_norm_epsilon"): "layernorm_epsilon",
    ("qkv_has_bias", "attention_bias", "add_qkv_bias"): "add_qkv_bias",
    ("expert_num", "n_routed_experts", "num_experts", "num_moe_experts"): "num_moe_experts",
    ("num_layers", "num_hidden_layers", "n_layer"): "num_layers",
    ("rope_interleave", "rotary_interleaved"): "rotary_interleaved",
    ("use_qk_norm", "qk_layernorm"): "qk_layernorm",
    # not changes
    "hidden_size": "hidden_size",
    "softmax_scale": "softmax_scale",
    "hidden_dropout": "hidden_dropout",
    "attention_dropout": "attention_dropout",
    "apply_residual_connection_post_layernorm": "apply_residual_connection_post_layernorm",
    "layernorm_zero_centered_gamma": "layernorm_zero_centered_gamma",
    "add_bias_linear": "add_bias_linear",
    "gated_linear_unit": "gated_linear_unit",
    "activation_func": "activation_func",
    "normalization": "normalization",
    "fused_norm": "fused_norm",
    "calculate_per_token_loss": "calculate_per_token_loss",
    "multi_latent_attention": "multi_latent_attention",
    "compute_dtype": "compute_dtype",
    "layernorm_compute_dtype": "layernorm_compute_dtype",
    "rotary_dtype": "rotary_dtype",
    "seq_length": "seq_length",
    "vocab_size": "vocab_size",
    "ignore_token_id": "ignore_token_id",
    "is_dynamic": "is_dynamic",
    "use_eod_reset": "use_eod_reset",
    "use_contiguous_weight_layout_attention": "use_contiguous_weight_layout_attention",
    "use_interleaved_weight_layout_mlp": "use_interleaved_weight_layout_mlp",
    "partial_rotary_factor": "partial_rotary_factor",
    "pre_process": "pre_process",
    "post_process": "post_process",
    "add_mlp_fc1_bias_linear": "add_mlp_fc1_bias_linear",
    "add_mlp_fc2_bias_linear": "add_mlp_fc2_bias_linear",

    # Flash Attention
    # not changes
    "use_flash_attention": "use_flash_attention",
    "attention_pre_tokens": "attention_pre_tokens",
    "attention_next_tokens": "attention_next_tokens",
    "rotary_seq_len_interpolation_factor": "rotary_seq_len_interpolation_factor",
    "use_rope_scaling": "use_rope_scaling",
    "rope_scaling": "rope_scaling",
    "rotary_cos_format": "rotary_cos_format",
    "input_layout": "input_layout",
    "sparse_mode": "sparse_mode",
    "use_alibi_mask": "use_alibi_mask",
    "use_attn_mask_compression": "use_attn_mask_compression",
    "use_eod_attn_mask_compression": "use_eod_attn_mask_compression",
    "use_attention_mask": "use_attention_mask",
    "use_ring_attention": "use_ring_attention",
    "fp16_lm_cross_entropy": "fp16_lm_cross_entropy",
    "untie_embeddings_and_output_weights": "untie_embeddings_and_output_weights",
    "hidden_act": "hidden_act",
    "mask_func_type": "mask_func_type",
    ("extend_method", "position_embedding_type"): "position_embedding_type",
    ("init_method_std", "initializer_range"): "init_method_std",

    # Initialization
    # not changes
    "init_method": "init_method",
    "output_layer_init_method": "output_layer_init_method",
    "init_model_with_meta_device": "init_model_with_meta_device",

    # Mixed-Precision
    ("softmax_compute_dtype", "attention_softmax_in_fp32"): (
        "attention_softmax_in_fp32", is_float_32
    ),
    # not changes
    "apply_query_key_layer_scaling": "apply_query_key_layer_scaling",
    "disable_bf16_reduced_precision_matmul": "disable_bf16_reduced_precision_matmul",

    # Fusion
    ("use_fused_rope", "apply_rope_fusion"): "apply_rope_fusion",
    ("use_fused_swiglu", "bias_swiglu_fusion"): "bias_swiglu_fusion",
    ("use_fused_ops_permute", "moe_permute_fusion"): "moe_permute_fusion",
    # not changes
    "bias_activation_fusion": "bias_activation_fusion",
    "masked_softmax_fusion": "masked_softmax_fusion",
    "persist_layer_norm": "persist_layer_norm",
    "memory_efficient_layer_norm": "memory_efficient_layer_norm",
    "bias_dropout_fusion": "bias_dropout_fusion",

    # Recompute
    "recompute": (
        "recompute", get_recompute
    ),
    # not changes
    "select_recompute": "select_recompute",
    "parallel_optimizer_comm_recompute": "parallel_optimizer_comm_recompute",
    "select_comm_recompute": "select_comm_recompute",
    "mp_comm_recompute": "mp_comm_recompute",
    "recompute_slice_activation": "recompute_slice_activation",
    "select_recompute_exclude": "select_recompute_exclude",
    "select_comm_recompute_exclude": "select_comm_recompute_exclude",

    # Moe
    ("moe_intermediate_size", "moe_ffn_hidden_size"): "moe_ffn_hidden_size",
    ("num_experts_chosen", "num_experts_per_tok", "moe_router_topk"): "moe_router_topk",
    ("n_group", "moe_router_num_groups"): "moe_router_num_groups",
    ("topk_group", "moe_router_group_topk"): "moe_router_group_topk",
    ("routed_scaling_factor", "moe_router_topk_scaling_factor"): "moe_router_topk_scaling_factor",
    ("use_gating_sigmoid", "scoring_func", "moe_router_score_function"): (
        "moe_router_score_function", is_use_gated_sigmoid
    ),
    ("router_dense_type", "moe_router_dtype"): "moe_router_dtype",
    ("balance_via_topk_bias", "moe_router_enable_expert_bias"): "moe_router_enable_expert_bias",
    ("topk_bias_update_rate", "moe_router_bias_update_rate"): "moe_router_bias_update_rate",
    ("use_gmm", "moe_grouped_gemm"): "moe_grouped_gemm",
    ("z_loss_factor", "moe_z_loss_coeff"): "moe_z_loss_coeff",
    ("capacity_factor", "moe_expert_capacity_factor"): "moe_expert_capacity_factor",
    ("enable_sdrop", "moe_token_drop_policy"): ("moe_token_drop_policy", get_drop_policy),
    ("enable_gmm_safe_tokens", "use_pad_tokens"): "use_pad_tokens",
    "moe_shared_expert_intermediate_size": "moe_shared_expert_intermediate_size",
    ("n_shared_experts", "shared_expert_num"): "shared_expert_num",
    ("aux_loss_types", "moe_router_load_balancing_type"): "moe_router_load_balancing_type",
    ("aux_loss_factors", "moe_aux_loss_coeff"): "moe_aux_loss_coeff",
    # not changes
    "moe_layer_freq": "moe_layer_freq",
    "first_k_dense_replace": "first_k_dense_replace",
    "moe_shared_expert_overlap": "moe_shared_expert_overlap",
    "moe_router_pre_softmax": "moe_router_pre_softmax",
    "moe_input_jitter_eps": "moe_input_jitter_eps",
    "moe_token_dispatcher_type": "moe_token_dispatcher_type",
    "group_wise_a2a": "group_wise_a2a",
    "moe_enable_deepep": "moe_enable_deepep",
    "moe_per_layer_logging": "moe_per_layer_logging",
    "moe_pad_expert_input_to_capacity": "moe_pad_expert_input_to_capacity",
    "moe_apply_probs_on_input": "moe_apply_probs_on_input",
    "comp_comm_parallel": "comp_comm_parallel",
    "comp_comm_parallel_degree": "comp_comm_parallel_degree",
    "norm_topk_prob": "norm_topk_prob",
    "use_fused_ops_topkrouter": "use_fused_ops_topkrouter",
    "use_shared_expert_gating": "use_shared_expert_gating",
    "topk_method": "topk_method",
    "npu_nums_per_device": "npu_nums_per_device",
    "callback_moe_droprate": "callback_moe_droprate",
    "moe_init_method_std": "moe_init_method_std",
    "moe_router_force_expert_balance": "moe_router_force_expert_balance",

    # Context Parallel
    # not changes
    "context_parallel_algo": ("cp_comm_type", get_cp_comm_type),
    "ulysses_degree_in_cp": "hierarchical_context_parallel_sizes",

    # MLATransformerConfig
    ("qk_nope_head_dim", "qk_head_dim"): "qk_head_dim",
    ("qk_rope_head_dim", "qk_pos_emb_head_dim"): "qk_pos_emb_head_dim",
    ("theta", "rope_theta", "rotary_base"): "rotary_base",
    ("scaling_factor", "factor", "rotary_scaling_factor"): "rotary_scaling_factor",
    ("max_position_embeddings", "original_max_position_embeddings"): "max_position_embeddings",
    # not changes
    "q_lora_rank": "q_lora_rank",
    "kv_lora_rank": "kv_lora_rank",
    "v_head_dim": "v_head_dim",
    "rotary_percent": "rotary_percent",
    "beta_fast": "beta_fast",
    "beta_slow": "beta_slow",
    "mscale": "mscale",
    "mscale_all_dim": "mscale_all_dim",
    "mla_qkv_concat": "mla_qkv_concat",

    # Inference Param
    "pad_token_id": "pad_token_id",
    "tie_word_embeddings": "tie_word_embeddings",
    "block_size": "block_size",
    "num_blocks": "num_blocks",
    "parallel_decoding_params": "parallel_decoding_params",
    "sandwich_norm": "sandwich_norm",
    "attn_reduce_scatter": "attn_reduce_scatter",
    "attn_allgather": "attn_allgather",
    "attn_allreduce": "attn_allreduce",
    "ffn_reduce_scatter": "ffn_reduce_scatter",
    "ffn_allgather": "ffn_allgather",
    "ffn_allreduce": "ffn_allreduce",
    "use_alltoall": "use_alltoall",
    "dispatch_global_max_bs": "dispatch_global_max_bs",
    "quantization_config": "quantization_config",

    # Pet
    "pet_config": "pet_config"
}


def convert_to_transformer_config(
        model_config: PretrainedConfig = None, is_mla_model: bool = False,
        additional_map: dict = None, not_convert_whitelist: set = None
) -> Union[TransformerConfig, MLATransformerConfig]:
    """
    Convert ModelConfig to TransFormerConfig.

    Args:
        model_config (PretrainedConfig): A model config of MindSpore Transformers.
        is_mla_model (bool): Whether the model is an MLA Model.
        additional_map (dict): Custom parameter mapping, that can be provided by the user.
            If not None, it will update the `COMMON_CONFIG_MAPPING` .
            And the configurations with the same name in the `COMMON_CONFIG_MAPPING`
            will be overwritten by `additional_map`.
        not_convert_whitelist (set): Whitelist of keys that will not be converted in the corresponding model.

    Returns:
        An instance of TransformerConfig. If it is an MLA model, then returns an instance of MLATransformerConfig.
    """
    # Check whether the type of `model_config` is legal
    if model_config is None or not isinstance(model_config, (dict, PretrainedConfig)):
        raise ValueError(f"The ModelConfig should be an instance of 'PretrainedConfig' or 'dict', "
                         f"but got '{type(model_config)}'.")

    if isinstance(model_config, PretrainedConfig):
        model_config = model_config.to_dict()

    # Get the `convert_map` and `reversed_map`
    convert_map = deepcopy(COMMON_CONFIG_MAPPING)
    convert_map = scatter_multi_mapping_keys_to_mapping(convert_map)
    convert_map = update_addtional_map_to_mapping(convert_map, additional_map)
    reversed_mapping = get_reversed_mapping(convert_map)

    # Get the `not_convert_whitelist`
    if not_convert_whitelist is None:
        not_convert_whitelist = set()
    not_convert_whitelist.update(DEFAULT_WHITE_KEY)
    logger.info(f"These Keys of this model will do not need to be mapped: {not_convert_whitelist}")

    # Record the new Config after conversion
    update_dict = {}
    # Record the keys of `model_config` outside the mapping rules in conversion
    not_convert_keys_list = []

    def mapping_config(key, value):
        """Map the model_config's key and add it to 'update_dict'."""
        mapping_key = convert_map[key]
        if not isinstance(mapping_key, str):
            (mapping_key, trans_func) = mapping_key
            value = trans_func(value)
        if mapping_key in update_dict.keys():
            raise KeyError(f"Multiple configurations provided for the same setting. "
                           f"Please check these conflicting configs: {list(reversed_mapping[mapping_key])}")
        update_dict[mapping_key] = value

    # Start converting parameters
    if 'parallel_config' in model_config:
        for parallel_key, parallel_value in model_config['parallel_config'].items():
            if parallel_key in convert_map.keys():
                mapping_config(parallel_key, parallel_value)
        model_config.pop('parallel_config')
    for model_config_key, model_config_value in model_config.items():
        if model_config_key in not_convert_whitelist:
            continue
        if model_config_key in convert_map.keys():
            mapping_config(model_config_key, model_config_value)
        else:
            not_convert_keys_list.append(model_config_key)

    # If there are any unconverted key values, print them out to inform the user to check the configuration
    if not_convert_keys_list:
        raise ValueError(f"Keys: {not_convert_keys_list} dose not be converted! "
                         f"Please check your config parameters.")

    # If it is an MLA model, use MLATransformerConfig for initialization
    if is_mla_model:
        mla_transformer_config = MLATransformerConfig(**update_dict)
        logger.info(f"The converted MLATransformerConfig is: \n{asdict(mla_transformer_config)}")
        return mla_transformer_config

    transform_config = TransformerConfig(**update_dict)
    logger.info(f"The converted TransformerConfig is: \n{asdict(transform_config)}")
    return transform_config
