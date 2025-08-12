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

from dataclasses import dataclass
from typing import Union

from mindspore import dtype

from mindformers.modules.transformer.transformer import (
    TransformerOpParallelConfig,
    default_transformer_config
)

ms_dtype_mapping = {
    # Common floating point numbers types
    "float64": dtype.float64,
    "fp64": dtype.float64,
    "float32": dtype.float32,
    "fp32": dtype.float32,
    "bfloat16": dtype.bfloat16,
    "bf16": dtype.bfloat16,
    "float16": dtype.float16,
    "fp16": dtype.float16,
    # Signed integer types
    "int8": dtype.int8,
    "int16": dtype.int16,
    "int32": dtype.int32,
    "int64": dtype.int64,
    # Unsigned integer types
    "uint8": dtype.uint8,
    "uint16": dtype.uint16,
    "uint32": dtype.uint32,
    "uint64": dtype.uint64,
    # Complex number types
    "complex64": dtype.complex64,
    "complex128": dtype.complex128
}


def convert_str_to_mstype(type_str) -> dtype:
    """
    Utils for convert type string to mstype.

    Args:
        type_str (Union[str, dtype]): A string describing the dtype, or mindspore.dtype.

    Returns:
        A dtype of `mindspore.dtype` .
    """
    if not isinstance(type_str, str):
        raise TypeError(f"The type of 'type_str' must 'string', but got '{type(type_str)}'.")

    if type_str in ms_dtype_mapping.keys():
        return ms_dtype_mapping[type_str]

    raise ValueError(f"The value of 'type_str' must be in {list(ms_dtype_mapping.keys())}, "
                     f"but got '{type_str}'.")


@dataclass
class MFModelConfig:
    """
    Configuration parameters specifically for MF model training and inference operations,
    which are separate from HuggingFace's standard parameter collection and TransformerConfig.
    """
    ########################################################
    # General Configuration Items For MindSpore Transformers
    ########################################################

    parallel_config: Union[dict, TransformerOpParallelConfig] = None
    """Configs which contains parallel settings."""

    pet_config: dict = None
    """Config for Pattern-Exploiting Training (PET)."""

    context_parallel_algo: str = "colossalai_cp"
    """
    Algorithm to use for context parallelism.
    Can be "colossalai_cp", "ulysses_cp" or "hybrid_cp".
    Only effective when context_parallel > 1
    """

    is_dynamic: bool = False
    """Whether model is dynamic shape."""

    compute_dtype: str = "bfloat16"
    """Linear layer compute dtype."""

    layernorm_compute_dtype: str = "float32"
    """LayerNorm compute dtype."""

    rotary_dtype: str = "float32"
    """Custom rotary position embedding compute dtype."""

    bias_swiglu_fusion: bool = False
    """If True, use fused swiglu kernel."""

    qk_layernorm: bool = False
    """Whether to apply `normalization` type of normalization to the query and key embeddings."""

    mla_qkv_concat: bool = True
    """If True, Multi Latent Attention computes q_compressed, k, kv_compressed in a single linear transformation;
    if False, computes them separately."""

    use_contiguous_weight_layout: bool = True
    """
    Determines the weight arrangement in SelfAttention's QKV linear projection. Only affects SelfAttention layers.

    When True (default):
        Uses contiguous layout: [Q_weights, K_weights, V_weights]
        - Computation: linear(input) -> split into Q, K, V tensors

    When False:
        Uses interleaved head layout: [Q_head0, K_head0, V_head0, Q_head1, ...]
        - Matches Megatron-LM's weight arrangement
        - Computation: linear(input)
                    -> reshape
                    -> split into Q, K, V tensors grouped by attention heads

    Note: This affects tensor memory layout but not mathematical equivalence.
    """

    normalization: str = "LayerNorm"
    """Which norm to use for normalization layers, valid options are `LayerNorm` and `RMSNorm`."""

    add_bias_linear: bool = True
    """
    Include a bias term in all linear layers
    (QKV projections, after core attention, and two in MLP layer).
    """

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    ################################################################
    # Flash Attention Configuration Items for MindSpore Transformers
    ################################################################

    use_flash_attention: bool = True
    """If true, use flash attention for the attention layer."""

    attention_pre_tokens: int = None
    """Pre-tokens for flash attention."""

    attention_next_tokens: int = None
    """Next tokens for flash attention."""

    rotary_seq_len_interpolation_factor: float = None
    """
    RoPE scaling used for linear interpolation of longer sequences.
    This value must be a floating point number greater than 1.0.
    """

    rope_scaling: dict = None
    """Dictionary containing the scaling configuration for the RoPE embeddings."""

    use_rope_scaling: bool = False
    """Whether to use RoPE scaling."""

    input_layout: str = "BNSD"
    """
    Specifies the layout of input query, key and value.
    The value can be "BSH", "BNSD", "SBH", "BSND" or "TND". "TND" is an experimental format.
    More details, please refer MindSpore API Document: mindspore.ops.flash_attention_score
    (https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.flash_attention_score.html)
    """

    sparse_mode: int = 0
    """
    Indicates sparse mode:
    - 0: Indicates the defaultMask mode. If attn_mask is not passed, the mask operation is not performed,
        and preTokens and nextTokens(internally assigned as INT_MAX) are ignored. If passed in, the full attn_mask
        matrix (S1 * S2) needs to be passed in, indicating that the part between preTokens and nextTokens needs to
        be calculated.
    - 1: Represents allMask, that is, passing in the complete attn_mask matrix.
    - 2: Representing the leftUpCausal mode corresponds to the lower triangle scenario divided by the left
        vertex, and the optimized attn_mask matrix (2048*2048) is required.
    - 3: Representing the rightDownCausal model corresponds to the lower triangle scene divided by the lower
        right vertex, and the optimized attn_mask matrix (2048*2048) is required.
    - 4: Represents the band scenario, that is, the part between counting preTokens and nextTokens, and the
        optimized attn_mask matrix (2048*2048) is required..
    - 5: Represents the prefix scenario, that is, on the basis of rightDownCasual, a matrix with length S1 and
        width N is added to the left side. The value of N is obtained by the new input prefix, and the N value of
        each Batch axis is different. Not implemented yet.
    - 6: Represents the global scenario, not implemented yet.
    - 7: Represents the dilated scenario, not implemented yet.
    - 8: Represents the block_local scenario, not implemented yet.
    """

    use_alibi_mask: bool = False
    """The value is True if alibi_mask is passed."""

    use_attn_mask_compression: bool = False
    """If true, use attention mask compression for the attention layer."""

    use_eod_attn_mask_compression: bool = False
    """If true, use end of sequence attention mask compression for the attention layer."""

    use_attention_mask: bool = True
    """If true, use attention mask for the attention layer."""

    use_ring_attention: bool = False
    """If true, use ring attention for the attention layer."""

    fp16_lm_cross_entropy: bool = False
    """If true, use fp16 for cross entropy loss calculation."""

    untie_embeddings_and_output_weights: bool = True
    """If true, untie the embeddings and output weights."""

    hidden_act: str = "gelu"
    """Activation function to use for the non-linearity in the MLP."""

    mask_func_type: str = "attn_mask_fill"
    """Mask function type to use for the attention layer."""

    use_fused_ops_permute: bool = False
    """If True, use fused ops for permutation."""

    ####################################################
    # MoE Configuration Items For MindSpore Transformers
    ####################################################
    comp_comm_parallel: bool = False
    """
    Whether to enable ffn compute and communication parallel,
    which can reduce pure communicattion time by splitting and overlapping compute and communication.
    """

    comp_comm_parallel_degree: int = 2
    """
    The split number of compute and communication.
    The larger the numbers,the more overlap there will be but will consume more memory.
    This parameter is effective only when comp_comm_parallel enable.
    """

    norm_topk_prob: bool = True
    """If True, use top-k probability for normalization."""

    use_fused_ops_topkrouter: bool = False
    """If True, use fused ops for top-k routing."""

    use_shared_expert_gating: bool = False
    """If True, use shared expert gating."""

    topk_method: str = "greedy"
    """Method to use for top-k routing."""

    enable_deredundency: bool = False
    """This parameter is used for inter-machine communication masking and performance optimization features."""

    npu_nums_per_device: int = 1
    """Set NPU ranks for each device."""

    use_pad_tokens: bool = False
    """If True, gmm pads an additional protection token to avoid 0-token calculation."""

    callback_moe_droprate: bool = False
    """Whether to print each expert's load information through callback."""

    moe_init_method_std: float = 0.01
    """Standard deviation of the zero mean normal for the MoE initialization method."""

    first_k_dense_replace: int = None
    r"""
    Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                      \--k dense layers--/
    """

    aux_loss_types: list = None
    """List of auxiliary loss types."""

    aux_loss_factors: list = None
    """List of auxiliary loss factors."""

    moe_router_enable_expert_bias: bool = False
    """
    TopK routing with dynamic per-expert bias in the aux-loss-free load balancing strategy.
    The routing decision is based on the sum of the routing scores and the expert bias.
    See https://arxiv.org/abs/2408.15664 for details.
    """


    ################################################
    # Training Parameters for MindSpore Transformers
    ################################################

    use_eod_reset: bool = False
    """Whether to use eod reset."""

    hidden_dropout: float = 0.0
    """Dropout probability for transformer hidden state."""

    residual_dtype: str = None
    """
    Data type computed in residual connections.
    It will be converted to `fp32_residual_connection` in `TransformerConfig`.
    """

    #################################################
    # Inference Parameters for MindSpore Transformers
    #################################################

    vocab_size: int = 128000
    """Vocabulary size of the model."""

    seq_length: int = 4096
    """Model Seq Length"""

    pad_token_id: int = 0
    """Model pad token id."""

    ignore_token_id: int = -100
    """Model ignore token id when training."""

    max_position_embeddings: int = 4096
    """Maximum sequence length that the model can handle."""

    sandwich_norm: bool = False  # None
    """Whether to apply `normalization` type of normalization to the transformer layer."""

    tie_word_embeddings: bool = False
    """Whether to share the input and output embedding weights."""

    block_size: int = 16
    """Size of each memory block used in PagedAttention."""

    num_blocks: int = 512
    """Size of each memory block used in PagedAttention."""

    parallel_decoding_params: dict = None
    """Parameters used when hardware decoding."""

    softmax_compute_dtype: str = 'float32'
    """Data type for computing softmax during attention computation."""

    post_process: bool = True
    """When using pipeline parallel, indicate whether it's the last stage."""

    dispatch_global_max_bs: int = 0
    """Maximum global batch size in MoE dispatch with AlltoAll"""

    attn_reduce_scatter: bool = False
    """Whether to enable attn_reduce_scatter"""

    attn_allgather: bool = False
    """Whether to enable attn_allgather"""

    attn_allreduce: bool = True
    """Whether to enable attn_allreduce"""

    ffn_reduce_scatter: bool = False
    """Whether to enable ffn_reduce_scatter"""

    ffn_allgather: bool = False
    """Whether to enable ffn_allgather"""

    ffn_allreduce: bool = True
    """Whether to enable ffn_allreduce"""

    use_alltoall: bool = False
    """Whether to enable use_alltoall"""

    def __post_init__(self):
        self.parallel_config = default_transformer_config

        if self.residual_dtype is None:
            self.residual_dtype = self.compute_dtype
