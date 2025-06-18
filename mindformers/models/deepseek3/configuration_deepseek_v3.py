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
"""DeepSeek3 Config API."""
__all__ = ['Deepseek3Config']

from typing import Optional, Union, List

from mindspore._checkparam import args_type_check

from mindformers.modules.transformer.transformer import (default_transformer_config,
                                                         TransformerOpParallelConfig)
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.utils import convert_mstype


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class Deepseek3Config(PretrainedConfig):

    """ Deepseek3 Model Config """

    model_type = "Deepseek3"

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig))
    def __init__(self,
                 vocab_size: int = 129280,
                 hidden_size: int = 7168,
                 intermediate_size: Optional[int] = 18432,
                 moe_intermediate_size: Optional[int] = 2048,
                 num_hidden_layers: int = 4,
                 num_attention_heads: int = 128,
                 num_key_value_heads: int = 128,
                 n_shared_experts: int = 1,
                 n_routed_experts: int = 256,
                 routed_scaling_factor: float = 2.5,
                 kv_lora_rank: int = 512,
                 q_lora_rank: int = 1536,
                 qk_rope_head_dim: int = 64,
                 v_head_dim: int = 128,
                 qk_nope_head_dim: int = 128,
                 n_group: int = 8,
                 topk_group: int = 4,
                 num_experts_per_tok: int = 8,
                 moe_layer_freq: Union[int, List[int]] = None,
                 norm_topk_prob: bool = True,
                 hidden_act: str = "silu",
                 max_position_embeddings: int = 4096,
                 rms_norm_eps: float = 1e-6,
                 seq_length: int = 2048,
                 pad_token_id: int = None,
                 bos_token_id: int = 0,
                 eos_token_id: int = 1,
                 normalization: str = "RMSNorm",
                 compute_dtype: str = "float16",
                 layernorm_compute_dtype: str = "float32",
                 softmax_compute_dtype: str = "float32",
                 rotary_dtype: str = "float32",
                 params_dtype: str = "float16",
                 residual_dtype: str = None,
                 add_qkv_bias: bool = False,
                 add_bias_linear: bool = False,
                 gated_linear_unit: bool = True,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 use_flash_attention: bool = True,
                 tie_word_embeddings: bool = False,
                 rope_theta: float = 10000.0,
                 attention_dropout: float = 0.0,
                 repetition_penalty: float = 1.0,
                 max_decode_length: int = 1024,
                 block_size: int = 16,
                 num_blocks: int = 512,
                 top_k: int = 5,
                 top_p: float = 1.0,
                 do_sample: bool = True,
                 parallel_decoding_params: dict = None,
                 moe_router_enable_expert_bias: bool = True,
                 moe_router_score_function: str = "sigmoid",
                 qk_layernorm: bool = True,
                 **kwargs):
        r"""
        Deepseek3 config class which defines the model size.

        Args:
            vocab_size (`int`, *optional*, defaults to 129280):
                Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
                `inputs_ids` passed when calling [`DeepseekV3Model`]
            hidden_size (`int`, *optional*, defaults to 4096):
                Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 11008):
                Dimension of the MLP representations.
            moe_intermediate_size (`int`, *optional*, defaults to 1407):
                Dimension of the MoE representations.
            num_hidden_layers (`int`, *optional*, defaults to 32):
                Number of hidden layers in the Transformer decoder.
            num_nextn_predict_layers (`int`, *optional*, defaults to 1):
                Number of nextn predict layers in the DeepSeekV3 Model.
            num_attention_heads (`int`, *optional*, defaults to 32):
                Number of attention heads for each attention layer in the Transformer decoder.
            n_shared_experts (`int`, *optional*, defaults to None):
                Number of shared experts, None means dense model.
            n_routed_experts (`int`, *optional*, defaults to None):
                Number of routed experts, None means dense model.
            routed_scaling_factor (`float`, *optional*, defaults to 1.0):
                Scaling factor or routed experts.
            topk_method (`str`, *optional*, defaults to `gready`):
                Topk method used in routed gate.
            n_group (`int`, *optional*, defaults to None):
                Number of groups for routed experts.
            topk_group (`int`, *optional*, defaults to None):
                Number of selected groups for each token(for each token,
                ensuring the selected experts is only within `topk_group` groups).
            num_experts_per_tok (`int`, *optional*, defaults to None):
                Number of selected experts, None means dense model.
            moe_layer_freq (`int`, *optional*, defaults to 1):
                The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
            first_k_dense_replace (`int`, *optional*, defaults to 0):
                Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                                \--k dense layers--/
            norm_topk_prob (`bool`, *optional*, defaults to False):
                Whether to normalize the weights of the routed experts.
            scoring_func (`str`, *optional*, defaults to 'softmax'):
                Method of computing expert weights.
            aux_loss_alpha (`float`, *optional*, defaults to 0.001):
                Auxiliary loss weight coefficient.
            seq_aux = (`bool`, *optional*, defaults to True):
                Whether to compute the auxiliary loss for each individual sample.
            num_key_value_heads (`int`, *optional*):
                This is the number of key_value heads that should be used to implement Grouped Query Attention. If
                `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
                `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
                converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should
                be constructed by meanpooling all the original heads within that group.
                For more details checkout [this paper](https://arxiv.org/pdf/2305.13245.pdf).
                If it is not specified, will default to `num_attention_heads`.
            hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
                The non-linear activation function (function or string) in the decoder.
            max_position_embeddings (`int`, *optional*, defaults to 2048):
                The maximum sequence length that this model might ever be used with.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            rms_norm_eps (`float`, *optional*, defaults to 1e-06):
                The epsilon used by the rms normalization layers.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the last key/values attentions (not used by all models). Only
                relevant if `config.is_decoder=True`.
            pad_token_id (`int`, *optional*):
                Padding token id.
            bos_token_id (`int`, *optional*, defaults to 1):
                Beginning of stream token id.
            eos_token_id (`int`, *optional*, defaults to 2):
                End of stream token id.
            pretraining_tp (`int`, *optional*, defaults to 1):
                Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
                document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
                necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
                issue](https://github.com/pytorch/pytorch/issues/76232).
            tie_word_embeddings (`bool`, *optional*, defaults to `False`):
                Whether to tie weight embeddings
            rope_theta (`float`, *optional*, defaults to 10000.0):
                The base period of the RoPE embeddings.
            attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
                Whether to use a bias in the query, key, value and output projection layers during self-attention.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the attention probabilities.
        """
        super(Deepseek3Config, self).__init__(**kwargs)
        # hf params
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.norm_topk_prob = norm_topk_prob
        self.max_position_embeddings = max_position_embeddings if max_position_embeddings else seq_length
        self.v_head_dim = v_head_dim
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_dropout = attention_dropout
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.moe_router_score_function = moe_router_score_function
        # common params
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        self.parallel_config = parallel_config
        self.seq_length = seq_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.normalization = normalization
        self.compute_dtype = convert_mstype(compute_dtype)
        self.layernorm_compute_dtype = convert_mstype(layernorm_compute_dtype)
        self.softmax_compute_dtype = convert_mstype(softmax_compute_dtype)
        self.rotary_dtype = convert_mstype(rotary_dtype)
        self.params_dtype = convert_mstype(params_dtype)
        if residual_dtype is not None:
            self.residual_dtype = convert_mstype(residual_dtype)
        else:
            self.residual_dtype = self.compute_dtype
        self.add_qkv_bias = add_qkv_bias
        self.add_bias_linear = add_bias_linear
        self.gated_linear_unit = gated_linear_unit
        self.use_flash_attention = use_flash_attention
        self.qk_layernorm = qk_layernorm
        # infer params
        self.repetition_penalty = repetition_penalty
        self.max_decode_length = max_decode_length
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.parallel_decoding_params = parallel_decoding_params
