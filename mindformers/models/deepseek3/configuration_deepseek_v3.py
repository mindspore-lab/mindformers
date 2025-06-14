# coding=utf-8
# Copyright 2025 bzantium and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on the DeepSeekV3 implementations from the DeepSeek AI team. (https://huggingface.co/deepseek-ai/DeepSeek-V3)
#
# Modification points:
# 1. Change `PretrainedConfig` to MindSpore Transformers;
# 2. Delete useless code for logging;
# 3. Add the `__all__` information of the Config class;
# 4. Add `MindFormerRegister` decorator to adapt to training/inference process of MindSpore Transformers;
# 5. Add `register_mf_model_parameter` decorator to pass other required parameters except HuggingFace parameters;
# 6. Add `ignore_and_delete_parameter` decorator to shield unnecessary configuration information.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeekV3 HuggingFace Model Configs."""

__all__ = ['DeepseekV3Config']

from typing import Union, List

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.parallel_core.mf_model_config import MFModelConfig
from mindformers.models.model_config_utils import (
    register_mf_model_parameter,
    ignore_and_delete_parameter,
    NotSupportedInfo
)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


@MindFormerRegister.register(MindFormerModuleType.CONFIG, legacy=False, search_names='deepseekv3')
class DeepseekV3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV3Model`].
    It is used to instantiate an DeepSeek model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the DeepSeek-V3.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (int, optional):
            Vocabulary size of the Deep model. Defines the number of different tokens that
            can be represented by the `inputs_ids` passed when calling [`DeepseekV3Model`]. Default: 129280.
        hidden_size (int, optional):
            Dimension of the hidden representations. Default: 4096.
        intermediate_size (int, optional):
            Dimension of the MLP representations. Default: 11008.
        moe_intermediate_size (int, optional):
            Dimension of the MoE representations. Default: 1407.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer decoder. Default: 32.
        num_nextn_predict_layers (int, optional):
            Number of nextn predict layers in the DeepSeekV3 Model. Default: 1.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer decoder. Default: 32.
        n_shared_experts (int, optional):
            Number of shared experts, None means dense model. Default: None.
        n_routed_experts (int, optional):
            Number of routed experts, None means dense model. Default: None.
        routed_scaling_factor (float, optional):
            Scaling factor or routed experts. Default: 1.0.
        topk_method (str, optional):
            Topk method used in routed gate. Default: 'greedy'.
        n_group (int, optional):
            Number of groups for routed experts. Default: None.
        topk_group (int, optional):
            Number of selected groups for each token. Default: None.
            (for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok (int, optional):
            Number of selected experts, None means dense model. Default: None.
        moe_layer_freq (int, optional):
            The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers. Default: 1.
        first_k_dense_replace (int, optional):
            Number of dense layers in shallow layers. Default: 0.
            (embed->dense->dense->...->dense->moe->moe...->lm_head)
                      \--k dense layers--/
        norm_topk_prob (bool, optional):
            Whether to normalize the weights of the routed experts. Default: `False`.
        scoring_func (str, optional):
            Method of computing expert weights. Default: 'softmax'.
        num_key_value_heads (int, optional):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by mean-pooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (str, optional):
            The non-linear activation function (function or string) in the decoder. Default: 'silu'.
        max_position_embeddings (int, optional):
            The maximum sequence length that this model might ever be used with. Default: 4096.
        initializer_range (float, optional):
            The standard deviation of the truncated_normal_initializer
            for initializing all weight matrices. Default: 0.02.
        rms_norm_eps (float, optional):
            The epsilon used by the rms normalization layers. Default: 1e-06.
        pad_token_id (int, optional):
            Padding token id.
        bos_token_id (int, optional):
            Beginning of stream token id. Default: 1.
        eos_token_id (int, optional):
            End of stream token id. Default: 2.
        tie_word_embeddings (bool, optional):
            Whether to tie weight embeddings. Default: `False`
        rope_theta (float, optional):
            The base period of the RoPE embeddings. Default: 10000.0.
        rope_scaling (Dict, optional):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently, supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{'type': strategy name, 'factor': scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        attention_bias (bool, optional):
            Whether to use a bias in the query, key, value layers during self-attention. Default: `False`.
        attention_dropout (float, optional):
            The dropout ratio for the attention probabilities. Default: 0.0.
        normalization (str, optional):
            normalization layer for MLA. Default: 'RMSNorm'.
        qk_layernorm (str, optional):
            Whether to apply `normalization` type of normalization to the query and key embeddings. Default: `True`.
        compute_dtype (str, optional): Linear layer compute dtype. Default: `float16`.
        layernorm_compute_type (str, optional): Layernorm compute dtype. Default: `float32`.
        softmax_compute_type (str, optional): Softmax compute dtype. Default: `float32`.
        rotary_dtype (str, optional): RoPE compute dtype. Default: `float32`.
        router_dense_type (str, optional): MoE router compute dtype. Default: `float32`.
        params_dtype (str, optional): Parameter initial dtype. Default: `float16`.
        residual_dtype (str, optional):
            Residual connections compute data type. Defaults: None, means the same as `compute_dtype`.
        gated_linear_unit (bool, optional):
            Whether to use a gated linear unit for the first linear layer in the MLP. Default: `False`.
        add_bias_linear (bool, optional):
            Whether to use a bias in all linear layers. Default
        hidden_dropout (float, optional):
            The dropout ratio for the transformer hidden state. Default: 0.0.
        moe_router_enable_expert_bias (bool, optional):
            Whether to use topk bias update. Default: `False`.
        moe_router_bias_update_rate (float, optional):
            How fast is the bias updated. Default: 0.001.
        moe_token_drop_policy (str, optional):
            The policy to drop tokens.
            If 'probs', the tokens with the lowest probabilities will be dropped.
            if 'position', tokens at the end of each batch will be dropped. Default: probs.
        aux_loss_factors (list, optional):
            list of auxiliary loss factors. Default: [0.0001].
        aux_loss_types (list, optional):
            list of auxiliary loss types. Default: ['expert'].
        moe_grouped_gemm (bool, optional):
            Whether to enable group matrix matmul. Should be set True manually currently. Default: False
    """

    model_type = "deepseekv3"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Use the decorators provided in MF to intercept unsupported and register MF custom parameters
    @register_mf_model_parameter(mf_model_kwargs=MFModelConfig(
        seq_length=4096,
        compute_dtype='bf16',
        layernorm_compute_dtype="fp32",
        softmax_compute_dtype="fp32",
        rotary_dtype="fp32",
        hidden_dropout=0.0,
        use_flash_attention=True,
        aux_loss_factors=[0.0001],
        aux_loss_types=['expert']
    ))
    @ignore_and_delete_parameter(extra_ignore_param=[
        ('ep_size', NotSupportedInfo.useless),
        ('quantization_config', NotSupportedInfo.not_implemented)
    ])
    def __init__(
            self,
            vocab_size: int = 129280,
            hidden_size: int = 7168,
            intermediate_size: int = 18432,
            moe_intermediate_size: int = 2048,
            num_hidden_layers: int = 61,
            num_nextn_predict_layers: int = 1,
            num_attention_heads: int = 128,
            num_key_value_heads: int = 128,
            n_shared_experts: int = 1,
            n_routed_experts: int = 256,
            ep_size: int = 1,
            routed_scaling_factor: float = 2.5,
            kv_lora_rank: int = 512,
            q_lora_rank: int = 1536,
            qk_rope_head_dim: int = 64,
            v_head_dim: int = 128,
            qk_nope_head_dim: int = 128,
            topk_method: str = 'noaux_tc',
            n_group: int = 8,
            topk_group: int = 4,
            num_experts_per_tok: int = 8,
            moe_layer_freq: Union[int, List[int]] = 1,
            first_k_dense_replace: int = 3,
            norm_topk_prob: bool = True,
            scoring_func: str = "sigmoid",
            hidden_act: str = "silu",
            max_position_embeddings: int = 4096,
            initializer_range: float = 0.02,
            rms_norm_eps: float = 1e-6,
            use_cache: bool = True,
            pad_token_id: int = None,
            bos_token_id: int = 0,
            eos_token_id: int = 1,
            tie_word_embeddings: bool = False,
            rope_theta: float = 10000.0,
            rope_scaling: dict = None,
            attention_bias: bool = False,
            attention_dropout: float = 0.0,
            **kwargs
    ):
        """Deepseek V3 Config"""
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size # useless in MindFormers
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
