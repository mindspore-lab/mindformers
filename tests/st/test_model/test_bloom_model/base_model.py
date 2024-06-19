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
"""Bloom Base Model."""
from mindformers.models.bloom import BloomLMHeadModel, BloomConfig

# copy from run_bloom_7.1b.yaml
BASE_CONFIG = {
    'attention_dropout_rate': 0.1,
    'bos_token_id': 1,
    'compute_dtype': 'float16',
    'embedding_init_type': 'float32',
    'eos_token_id': 2,
    'expand_ratio': 4,
    'hidden_act': 'gelu',
    'hidden_dropout_rate': 0.1,
    'hidden_size': 4096,
    'initializer_range': 0.02,
    'layernorm_compute_type': 'float32',
    'max_decode_length': 1024,
    'num_heads': 32,
    'num_layers': 2,  # 30
    'param_init_type': 'float16',
    'repetition_penalty': 1,
    'seq_length': 2048,
    'softmax_compute_type': 'float16',
    'top_k': 5,
    'top_p': 1,
    'type': 'BloomConfig',
    'use_flash_attention': True,
    'use_select_recompute': False,
    'use_seq_parallel': True,
    'vocab_size': 250880
}


def get_config():
    """get instanced model config."""
    return BloomConfig(**BASE_CONFIG)


def get_model(config):
    """get instanced model."""
    return BloomLMHeadModel(config)
