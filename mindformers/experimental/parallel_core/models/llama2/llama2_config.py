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
"""Llama2 Model config"""

from mindformers.experimental.parallel_core.pynative.config import TransformerConfig
from mindformers.experimental.parallel_core.pynative.utils import DictWithValueError


COMMON_CONFIG = {
    "vocab_size": 32000,
    "seq_length": 4096,
    "position_embedding_type": "rope",
    # attention
    "attention_type": "self_attn",
    "encoder_attn_mask_type": None,
    "apply_residual_connection_post_norm": False,
    "use_flash_attention": True,
    "qkv_has_bias": False,
    "out_proj_has_bias": False,
    "mask_func_type": "attn_mask_fill",
    "softmax_compute_dtype": "float32",
    # layer norm
    "normalization": "FusedRMSNorm",
    "norm_epsilon": 1e-5,
    # hidden_act
    "hidden_act": "silu",
    # dropout
    "hidden_dropout": 0.1,
    "attention_dropout": 0.1,
    # mlp
    "mlp_has_bias": False,
    "mlp_has_gate": True,
    "num_experts": None,
    # shared weights
    "head_skip_weight_param_allocation": True,
    "post_norm": True,
    "fa_config": {
        "input_layout": "BNSD"
    }
}

MODEL_CONFIGS = DictWithValueError({
    "7B": {
        "hidden_size": 4096,
        "ffn_hidden_size": 11008,
        "num_layers": 32,
        "num_attention_heads": 32,
    },
    "13B": {
        "hidden_size": 5120,
        "ffn_hidden_size": 13824,
        "num_layers": 40,
        "num_attention_heads": 40,
    },
    "70B": {
        "hidden_size": 8192,
        "ffn_hidden_size": 28672,
        "num_layers": 80,
        "num_attention_heads": 64,
        "group_query_attention": True,
        "num_query_groups": 8,
    },
})


class Llama2Config(TransformerConfig):
    """Pangu Model Config"""

    config_name = "llama2_config"

    def __init__(self, model_type="7B", **kwargs):
        config = COMMON_CONFIG.copy()
        config.update(MODEL_CONFIGS[model_type])
        config.update(kwargs)
        super(Llama2Config, self).__init__(**config)

        self.model_type = model_type
