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
"""Deepseekv3 Base Model."""
from research.deepseek3.deepseek3_config import DeepseekV3Config
from research.deepseek3.deepseek3_model_train import TrainingDeepseekV3ForCausalLM


# modify from pretrain_deepseek3_671b.yaml
MOE_CONFIG = {
    'expert_num': 4,  # origin 256
    'expert_group_size': 8,
    'capacity_factor': 1.5,
    'aux_loss_factor': 0.05,
    'num_experts_chosen': 2,  # origin 8
    'routing_policy': "TopkRouterV2",
    'balance_via_topk_bias': True,
    'topk_bias_update_rate': 0.001,
    'use_fused_ops_topkrouter': True,
    'shared_expert_num': 1,
    'routed_scaling_factor': 2.5,
    'norm_topk_prob': True,
    'first_k_dense_replace': 1,  # origin 3
    'moe_intermediate_size': 2048,
    'aux_loss_factors': [0.0001],
    'aux_loss_types': ["expert"],
    'expert_model_parallel': 1,
    'use_gating_sigmoid': True,
    'callback_moe_droprate': False,
    'use_gmm': True,
    'use_fused_ops_permute': True,
    'enable_gmm_safe_tokens': True,
    'enable_deredundency': False,
    'npu_nums_per_device': 2
}


# modify from pretrain_deepseek3_671b.yaml
BASE_CONFIG = {
    'batch_size': 1,
    'hidden_size': 2048,  # origin 7168
    'num_layers': 3,  # origin 61
    'num_heads': 16,  # origin 128
    'max_position_embeddings': 4096,
    'intermediate_size': 4096,  # origin 18432
    'kv_lora_rank': 512,
    'n_kv_heads': 128,
    'q_lora_rank': 1536,
    'qk_rope_head_dim': 64,
    'v_head_dim': 128,
    'qk_nope_head_dim': 128,
    'vocab_size': 12000,  # origin 129280
    'multiple_of': 256,
    'rms_norm_eps': 1.0e-6,
    'compute_dtype': "bfloat16",
    'layernorm_compute_type': "float32",
    'softmax_compute_type': "float32",
    'rotary_dtype': "float32",
    'router_dense_type': "float32",
    'param_init_type': "float32",
    'extend_method': "None",
    'use_flash_attention': True,
    'use_fused_swiglu': True,
    'enable_fa_var_len': True,
    'use_fused_rope': True,
    'input_sliced_sig': False,
    'offset': 0,
    'checkpoint_name_or_path': "",
    'theta': 10000.0,
    'return_extra_loss': True,
    'mtp_depth': 1,
    'mtp_loss_factor': 0.3,
}


def get_config(model_config: dict = None):
    """get instanced model config."""
    base_config = BASE_CONFIG
    moe_config = MOE_CONFIG
    # replace the parameters in BASE_CONFIG or MOE_CONFIG
    if model_config:
        for key, value in model_config.items():
            if key in base_config:
                base_config[key] = value
            elif key in moe_config:
                moe_config[key] = value

    return DeepseekV3Config(**base_config, moe_config=moe_config)


def get_model(config):
    """get instanced model."""
    return TrainingDeepseekV3ForCausalLM(config)
