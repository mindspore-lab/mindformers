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
"""Pangu Model config"""

from mindformers.experimental.parallel_core.pynative.config import TransformerConfig
from mindformers.experimental.parallel_core.pynative.utils import DictWithValueError, calculate_dividable_vocab_size


COMMON_CONFIG = {
    "vocab_size": calculate_dividable_vocab_size(50257),
    # attention
    "group_query_attention": False,
    "apply_residual_connection_post_norm": False,
    "use_flash_attention": False,
    "qkv_has_bias": True,
    "out_proj_has_bias": True,
    "mask_func_type": "attn_mask_fill",
    # layer norm
    "normalization": "LayerNorm",
    "norm_epsilon": 1e-5,
    "hidden_act": "fast_gelu",
    # dropout
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
    # mlp
    "mlp_has_bias": True,
    # shared weights
    "head_skip_weight_param_allocation": True,
}

MODEL_CONFIGS = DictWithValueError({
    "one_layer": {
        "hidden_size": 2560,
        "num_layers": 2,  # one for query layer
        "num_attention_heads": 32,
        "num_experts": None,
    },
    "eight_layer": {
        "hidden_size": 2560,
        "num_layers": 8,
        "num_attention_heads": 32,
        "num_experts": None,
    },
    "2.6B": {
        "hidden_size": 2560,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_experts": None,
    },
    "135B": {
        "hidden_size": 10240,
        "num_layers": 108,
        "num_attention_heads": 40,
        "num_experts": None,
    },
})


class PanguConfig(TransformerConfig):
    """Pangu Model Config"""

    config_name = "pangu_config"

    def __init__(self, model_type="one_layer", **kwargs):
        config = COMMON_CONFIG.copy()
        config.update(MODEL_CONFIGS[model_type])
        config.update(kwargs)
        config['ffn_hidden_size'] = 4 * config['hidden_size']
        super(PanguConfig, self).__init__(**config)

        self.model_type = model_type
