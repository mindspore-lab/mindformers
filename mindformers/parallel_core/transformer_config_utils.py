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

from copy import deepcopy

from mindformers import PretrainedConfig
from mindformers.parallel_core.transformer_config import TransformerConfig, MLATransformerConfig
from mindformers.tools.logger import logger


def is_float_32(dtype: str) -> bool:
    """
    This method to check whether the dtype is float32.

    Args:
        dtype (str): A string of dtype.

    Returns:
        A bool flag.
        If the dtype satisfies the fp32 validation requirements, return `True`. Otherwise, return 'False'.
    """
    if dtype in ["fp32", "float32"]:
        return True

    return False


def get_moe_layer_freq(value: int) -> dict:
    """
    This method to check parameter types of 'first_k_dense_replace'.

    Args:
        value (int): The value of 'first_k_dense_replace'. Currently only supports 'int'.

    Returns:
        The value of 'first_k_dense_replace'. If the type is 'int'.
    """
    if not isinstance(value, int):
        raise ValueError(f"The type of key 'first_k_dense_replace' is 'int', but got '{type(value)}'.")

    return value


def is_use_gated_sigmod(used_gated_sigmod: bool) -> str:
    """
    This method to check whether to use gated 'sigmod' or 'softmax'.

    Args:
        used_gated_sigmod (bool): A flag that indicate whether to use gated sigmod.

    Returns:
        A string to choose the function for MoE router score.
    """
    if used_gated_sigmod:
        return "sigmod"

    return "softmax"


COMMON_CONFIG_MAPPING = {
    # ModelParallelConfig
    # Model parallelism
    "data_parallel": "data_parallel_size",
    "model_parallel": "tensor_model_parallel_size",
    "pipeline_stage": "pipeline_model_parallel_size",
    "pp_interleave_num": "virtual_pipeline_model_parallel_size",
    "use_seq_parallel": "sequence_parallel",
    "context_parallel": "context_parallel_size",
    "expert_parallel": "expert_model_parallel_size",
    "expert_model_parallel": "expert_tensor_parallel_size",
    # not changes
    "hierarchical_context_parallel_sizes": "hierarchical_context_parallel_sizes",
    "micro_batch_num": "micro_batch_num",
    "seq_split_num": "seq_split_num",
    "gradient_aggregation_group": "gradient_aggregation_group",
    "offset": "offset",
    "ulysses_degree_in_cp": "ulysses_degree_in_cp",
    "vocab_emb_dp": "vocab_emb_dp",

    # Training
    "param_init_type": "params_dtype",
    # not changes
    "finalize_model_grads_func": "finalize_model_grads_func",
    "grad_scale_func": "grad_scale_func",
    "grad_sync_func": "grad_sync_func",
    "param_sync_func": "param_sync_func",
    "num_microbatches_with_partial_activation_checkpoints": "num_microbatches_with_partial_activation_checkpoints",

    # CPU Offloading
    "swap": "cpu_offloading",
    "layer_swap": "cpu_offloading_num_layers",
    # not changes
    "op_swap": "op_swap",
    "default_prefetch": "default_prefetch",

    # TransformerConfig
    # Model Architecture
    "mtp_depth": "mtp_num_layers",
    "mtp_loss_factor": "mtp_loss_scaling_factor",
    "num_heads": "num_attention_heads",
    "n_kv_heads": "num_query_groups",
    "intermediate_size": "ffn_hidden_size",
    "head_dim": "kv_channels",
    "residual_dtype": ("fp32_residual_connection", is_float_32),
    "rms_norm_eps": "layernorm_epsilon",
    "qkv_has_bias": "add_qkv_bias",
    "expert_num": "num_moe_experts",
    # not changes
    "num_layers": "num_layers",
    "hidden_size": "hidden_size",
    "softmax_scale": "softmax_scale",
    "hidden_dropout": "hidden_dropout",
    "attention_dropout": "attention_dropout",
    "apply_residual_connection_post_layernorm": "apply_residual_connection_post_layernorm",
    "layernorm_zero_centered_gamma": "layernorm_zero_centered_gamma",
    "add_bias_linear": "add_bias_linear",
    "gated_linear_unit": "gated_linear_unit",
    "activation_func": "activation_func",
    "rotary_interleaved": "rotary_interleaved",
    "normalization": "normalization",
    "qk_layernorm": "qk_layernorm",
    "calculate_per_token_loss": "calculate_per_token_loss",
    "multi_latent_attention": "multi_latent_attention",
    "compute_dtype": "compute_dtype",
    "layernorm_compute_dtype": "layernorm_compute_dtype",
    "rotary_dtype": "rotary_dtype",

    # Flash Attention
    # not changes
    "use_flash_attention": "use_flash_attention",
    "attention_pre_tokens": "attention_pre_tokens",
    "attention_next_tokens": "attention_next_tokens",
    "rotary_seq_len_interpolation_factor": "rotary_seq_len_interpolation_factor",
    "rope_scaling": "rope_scaling",
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
    "position_embedding_type": "position_embedding_type",

    # Initialization
    # not changes
    "init_method": "init_method",
    "output_layer_init_method": "output_layer_init_method",
    "init_method_std": "init_method_std",
    "init_model_with_meta_device": "init_model_with_meta_device",

    # Mixed-Precision
    "softmax_compute_dtype": ("attention_softmax_in_fp32", is_float_32),
    # not changes
    "apply_query_key_layer_scaling": "apply_query_key_layer_scaling",
    "disable_bf16_reduced_precision_matmul": "disable_bf16_reduced_precision_matmul",

    # Fusion
    "use_fused_rope": "apply_rope_fusion",
    # not changes
    "bias_activation_fusion": "bias_activation_fusion",
    "masked_softmax_fusion": "masked_softmax_fusion",
    "persist_layer_norm": "persist_layer_norm",
    "memory_efficient_layer_norm": "memory_efficient_layer_norm",
    "bias_dropout_fusion": "bias_dropout_fusion",

    # Recompute
    # not changes
    "recompute": "recompute",
    "select_recompute": "select_recompute",
    "parallel_optimizer_comm_recompute": "parallel_optimizer_comm_recompute",
    "select_comm_recompute": "select_comm_recompute",
    "mp_comm_recompute": "mp_comm_recompute",
    "recompute_slice_activation": "recompute_slice_activation",
    "select_recompute_exclude": "select_recompute_exclude",
    "select_comm_recompute_exclude": "select_comm_recompute_exclude",

    # Moe
    "moe_intermediate_size": "moe_shared_expert_intermediate_size",
    "first_k_dense_replace": ("moe_layer_freq", get_moe_layer_freq),
    "num_experts_chosen": "moe_router_topk",
    "n_group": "moe_router_num_groups",
    "topk_group": "moe_router_group_topk",
    "routed_scaling_factor": "moe_router_topk_scaling_factor",
    "use_gated_sigmod": ("moe_router_score_function", is_use_gated_sigmod),
    "router_dense_type": "moe_router_dtype",
    "balance_via_topk_bias": "moe_router_enable_expert_bias",
    "topk_bias_update_rate": "moe_router_bias_update_rate",
    "use_gmm": "moe_grouped_gemm",
    "z_loss_factor": "moe_z_loss_coeff",
    "capacity_factor": "moe_expert_capacity_factor",
    "enable_sdrop": "moe_token_drop_policy",
    # not changes
    "moe_shared_expert_overlap": "moe_shared_expert_overlap",
    "moe_ffn_hidden_size": "moe_ffn_hidden_size",
    "moe_router_load_balancing_type": "moe_router_load_balancing_type",
    "moe_router_pre_softmax": "moe_router_pre_softmax",
    "aux_loss_types": "aux_loss_types",
    "aux_loss_factors": "aux_loss_factors",
    "moe_input_jitter_eps": "moe_input_jitter_eps",
    "use_allgather_dispatcher": "use_allgather_dispatcher",
    "group_wise_a2a": "group_wise_a2a",
    "moe_enable_deepep": "moe_enable_deepep",
    "moe_per_layer_logging": "moe_per_layer_logging",
    "moe_pad_expert_input_to_capacity": "moe_pad_expert_input_to_capacity",
    "moe_permute_fusion": "moe_permute_fusion",
    "moe_apply_probs_on_input": "moe_apply_probs_on_input",
    "comp_comm_parallel": "comp_comm_parallel",
    "comp_comm_parallel_degree": "comp_comm_parallel_degree",
    "norm_topk_prob": "norm_topk_prob",
    "use_fused_ops_topkrouter": "use_fused_ops_topkrouter",
    "shared_expert_num": "shared_expert_num",
    "use_shared_expert_gating": "use_shared_expert_gating",
    "topk_method": "topk_method",
    "enable_deredundency": "enable_deredundency",
    "npu_nums_per_device": "npu_nums_per_device",
    "enable_gmm_safe_tokens": "enable_gmm_safe_tokens",
    "use_fused_ops_permute": "use_fused_ops_permute",
    "callback_moe_droprate": "callback_moe_droprate",
    "return_extra_loss": "return_extra_loss",
    "moe_init_method_std": "moe_init_method_std",

    # Context Parallel
    # not changes
    "cp_comm_type": "cp_comm_type",
    "context_parallel_algo": "context_parallel_algo",

    # MLATransformerConfig
    "q_lora_rank": "q_lora_rank",
    "kv_lora_rank": "kv_lora_rank",
    "qk_head_dim": "qk_head_dim",
    "qk_pos_emb_head_dim": "qk_pos_emb_head_dim",
    "v_head_dim": "v_head_dim",
    "rope_type": "rope_type",
    "rotary_base": "rotary_base",
    "rotary_percent": "rotary_percent",
    "rotary_scaling_factor": "rotary_scaling_factor",
    "max_position_embeddings": "max_position_embeddings",
    "beta_fast": "beta_fast",
    "beta_slow": "beta_slow",
    "mscale": "mscale",
    "mscale_all_dim": "mscale_all_dim"
}


def convert_to_transformer_config(model_config: PretrainedConfig = None,
                                  is_mla_model: bool = False, additional_map: dict = None,
                                  not_convert_whitelist: list = None) -> TransformerConfig:
    """
    Convert ModelConfig to TransFormerConfig.

    Args:
        model_config (PretrainedConfig): A model config of MindSpore Transformers.
        is_mla_model (bool): Whether the model is an MLA Model.
        additional_map (dict): Custom parameter mapping, that can be provided by the user.
            If not None, it will update the `COMMON_CONFIG_MAPPING` .
            And the configurations with the same name in the `COMMON_CONFIG_MAPPING`
            will be overwritten by `additional_map`.
        not_convert_whitelist (list): Whitelist of keys that will not be converted in the corresponding model.

    Returns:
        An instance of TransformerConfig. If it is an MLA model, then returns an instance of MLATransformerConfig.
    """
    if model_config is None:
        raise ValueError(f"The ModelConfig should be an instance of 'PretrainedConfig', but got {model_config}")

    convert_map = deepcopy(COMMON_CONFIG_MAPPING)
    if additional_map is not None:
        convert_map.update(additional_map)

    if not_convert_whitelist is not None:
        logger.info(f"These Keys of this model will do not need to be mapped: {not_convert_whitelist}")

    # Convert Config Keys
    update_dict = {}
    not_convert_keys_list = []

    for model_config_key, model_config_value in model_config.items():
        if not_convert_whitelist and model_config_key in not_convert_whitelist:
            continue
        elif model_config_key in convert_map.keys():
            mapping_key = convert_map[model_config_key]
            if isinstance(mapping_key, str):
                update_dict[mapping_key] = model_config_value
            else:
                (transformer_config_key, trans_func) = mapping_key
                update_dict[transformer_config_key] = trans_func(model_config_value)
        else:
            not_convert_keys_list.append(model_config_key)

    if not_convert_keys_list:
        raise ValueError(f"Keys: {not_convert_keys_list} dose not be converted! "
                         f"Please check your config parameters.")

    # If it is an MLA model, use MLATransformerConfig for initialization
    if is_mla_model:
        logger.info(f"The converted MLATransformerConfig is: \n{MLATransformerConfig(**update_dict)}")
        return MLATransformerConfig(**update_dict)

    logger.info(f"The converted TransformerConfig is: \n{TransformerConfig(**update_dict)}")
    return TransformerConfig(**update_dict)
