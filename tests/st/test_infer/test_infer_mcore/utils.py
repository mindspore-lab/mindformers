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
"""Qwen2 Base Model."""


QWEN2_CONFIG = {
    "batch_size": 1,
    "seq_length": 8192,
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "intermediate_size": 4864,
    "max_position_embeddings": 32768,
    "qkv_has_bias": True,
    "rms_norm_eps": 1.0e-6,
    "rope_theta": 1000000.0,
    "emb_dropout_prob": 0.0,
    "compute_dtype": "bfloat16",
    "layernorm_compute_type": "float32",
    "softmax_compute_type": "float32",
    "rotary_dtype": "bfloat16",
    "param_init_type": "bfloat16",
    "use_flash_attention": True,
    "block_size": 32,
    "num_blocks": 1024,
    "qkv_concat": True,
    "tie_word_embeddings": True,
    "normalization": "RMSNorm",
    "hidden_act": "silu",
    "attn_proj_has_bias": False,
    "out_proj_has_bias": False,
    "mlp_has_bias": False
}

QWEN2_GENERATION_CONFIG = {
    "repetition_penalty": 1.1,
    "max_decode_length": 512,
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.8,
    "do_sample": False,
    "eos_token_id": [151643, 151645],
    "pad_token_id": 151643,
    "bos_token_id": 151643
}

def get_qwen_model_config():
    return QWEN2_CONFIG

def get_qwen_generation_config():
    return QWEN2_GENERATION_CONFIG
