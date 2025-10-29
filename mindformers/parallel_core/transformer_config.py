# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Modified some config parameters to adapt to MindSpore Transformer.
"""Transformer Config"""

import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import mindspore as ms
from mindformers.tools.logger import logger
from mindformers.parallel_core.mf_model_config import MFModelConfig, convert_str_to_mstype
from mindformers.parallel_core.model_parallel_config import ModelParallelConfig
from mindformers.parallel_core.utils.init_method import init_method_normal, scaled_init_method_normal


@dataclass
class TransformerConfig(ModelParallelConfig, MFModelConfig):
    """
    Configuration object for MindSpore Transformer's transformers.

    The initialization function has an argument for each parameter, including those in ModelParallelConfig.
    """

    ####################
    # Model Architecture
    ####################

    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    mtp_num_layers: Optional[int] = None
    """Number of Multi-Token Prediction (MTP) Layers."""

    mtp_loss_scaling_factor: Optional[float] = None
    """Weighting factor of Multi-Token Prediction (MTP) loss."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    softmax_scale: Optional[float] = None
    """Softmax scale for attention scaling."""

    num_query_groups: Optional[int] = None
    """Number of query groups for group query attention. If None, normal attention is used."""

    ffn_hidden_size: Optional[int] = None
    """
    Transformer Feed-Forward Network hidden size.
    This is set to 4*hidden_size if not provided.
    """

    kv_channels: Optional[int] = None
    """
    Projection weights dimension in multi-head attention.
    This is set to hidden_size // num_attention_heads if not provided.
    """

    hidden_dropout: float = 0.1
    """Dropout probability for transformer hidden state."""

    attention_dropout: float = 0.1
    """Post attention dropout probability."""

    fp32_residual_connection: bool = False
    """If true, move residual connections to fp32."""

    apply_residual_connection_post_layernorm: bool = False
    """If True, uses the original BERT residule connection ordering."""

    layernorm_epsilon: float = 1e-5
    """Epsilon value for any LayerNorm operations."""

    layernorm_zero_centered_gamma: bool = False
    """
    If set to True, the LayerNorm is adjusted to center the gamma values around 0.
    This improves numerical stability.
    """

    add_qkv_bias: bool = False
    """Add a bias term only for QKV projections."""

    activation_func: str = "gelu"
    """Activation function to use for the non-linearity in the MLP."""

    num_moe_experts: Optional[int] = None
    """
    Number of experts to use for MoE layer.
    When set, it replaces MLP with MoE layer. Set to None for no MoE.
    """

    rotary_interleaved: bool = False
    """
    True is rotate pairs of even and odd dimensions (RoFormer style),
    False is rotate pairs of first half and second half (LLaMa style). Default to False.
    """

    calculate_per_token_loss: bool = False
    """
    Whether cross entropy loss is calculated over the actual number of non-padded tokens in the
    global batch, versus the default behavior of assuming all tokens are non-padded.
    """

    multi_latent_attention: bool = False
    """Whether to use multi-latent attention."""

    position_embedding_type: str = "rope"
    """Position embedding type to use for the attention layer."""

    rotary_base: float = 10000.0
    """Rotary base for the rotary embeddings, used by rope and yarn. Mindformers required."""

    partial_rotary_factor: float = 1.0
    """rotaty partial dim"""

    qk_layernorm: bool = False
    """Whether to apply `normalization` type of normalization to the query and key embeddings."""

    ####################
    # Initialization
    ####################

    init_method: Optional[Callable] = None
    """
    Method to initialize weights. Note that bias is always set to zero.
    Should be a function that takes a single Tensor and initializes it.
    If None, will be set to init_method_normal(init_method_std)
    which is torch nn init normal with mean=0.0 and std=init_method_std.
    """

    output_layer_init_method: Optional[Callable] = None
    """
    Method to initialize weights of the output layer of both attention and MLP blocks.
    If None, will be set to scaled_init_method_normal(init_method_std)
    which is torch nn init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers).
    """

    init_method_std: float = 0.02
    """
    Standard deviation of the zero mean normal for the default initialization method,
    not used if init_method and output_layer_init_method are provided.
    """

    param_init_std_rules: List[dict[str, Union[str, float]]] = None
    """Configuration for decoupled weight initialization."""

    init_model_with_meta_device: bool = False
    """
    If True, initializes the model with the meta device. This is helpful for
    training of very large models. This feature is only works when custom fsdp is turned on.
    """

    ####################
    # Mixed-Precision
    ####################

    apply_query_key_layer_scaling: bool = False
    """
    If true, scale Q * K^T by 1 / layer-number.
    This improve numeric stability when training with fp16.
    """

    attention_softmax_in_fp32: bool = True
    """
    If True, run attention masking and softmax in fp32.
    This should be True if apply_query_key_layer_scaling is True.
    """

    softmax_compute_dtype: str = 'float32'
    """Data type for computing softmax during attention computation."""

    disable_bf16_reduced_precision_matmul: bool = False
    """If True, prevent matmul from using reduced precision accumulation when using BF16."""

    ####################
    # Fusion
    ####################

    bias_activation_fusion: bool = False
    """If True, fuses bias addition and the activation function when possible."""

    masked_softmax_fusion: bool = False
    """If True, uses softmax fusion."""

    persist_layer_norm: bool = False
    """
    If True, uses the persistent fused layer norm kernel.
    This kernel only supports a fixed set of hidden sizes.
    """

    memory_efficient_layer_norm: bool = False
    """
    If True, and using local layers (not from TransformerEngine),
    tells Apex to use the memory efficient fused LayerNorm kernel.
    Ignored if not using LayerNorm.
    """

    bias_dropout_fusion: bool = False
    """If True, uses bias dropout fusion."""

    apply_rope_fusion: bool = False
    """If True, use fused RoPE kernel."""

    ####################
    # Recompute
    ####################

    recompute: Optional[Union[bool, list, tuple]] = False
    """Whether enable recompute. Default: False."""

    select_recompute: Optional[Union[bool, list]] = False
    """Turn on recomputation to recompute only for the operators in the attention layer. Default: False."""

    parallel_optimizer_comm_recompute: Optional[bool] = False
    """Whether to recompute AllGather communication introduced in parallel by the optimizer. Default: False."""

    select_comm_recompute: Optional[bool] = False
    """Whether to slice the Cell outputs retained in memory. Default: False."""

    mp_comm_recompute: Optional[bool] = True
    """Whether to recompute communications introduced by model parallel. Default: True."""

    recompute_slice_activation: bool = False
    """Whether to output slices for Cells kept in memory. Default: False."""

    select_recompute_exclude: Optional[Union[bool, list]] = False
    """Disable recomputation for the specified operator, valid only for the Primitive operators."""

    select_comm_recompute_exclude: Optional[Union[bool, list]] = False
    """Disable communication recomputation for the specified operator, valid only for the Primitive operators."""

    ####################
    # MoE
    ####################

    moe_shared_expert_intermediate_size: Optional[int] = None
    """
    Shared expert total ffn hidden size.
    It should be equal to 'num_shared_experts * ffn_size_of_each_shared_expert' if
    there are multiple shared experts.
    None means no shared expert.
    """

    moe_shared_expert_overlap: bool = False
    """
    Enable overlapping between shared expert computations and dispatcher communications.
    Without this, the shared epxerts execute after the routed experts.
    """

    moe_layer_freq: Optional[Union[int, List[int]]] = 1
    """
    Frequency between MoE layers and Dense layers. Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers.
    - A list that defines a custom pattern, e.g.: [1,1,1,0,1,1,1,0,1,1,1,0]
    """

    moe_ffn_hidden_size: Optional[int] = None
    """MoE Feed-Forward Network hidden size"""

    moe_router_load_balancing_type: str = "sub_seq_aux_loss"
    """
    The load balancing strategy for the router.
    - "sub_seq_aux_loss" corresponds to the load balancing loss used in Legacy mode,
    - "seq_aux_loss" corresponds to the load balancing loss used in DeepSeekV2 and DeepSeekV3,
        which computes the loss for each individual sample;
    - "gbs_aux_loss" corresponds to the load balancing loss used in Qwen3MoE.

    The default is "sub_seq_aux_loss".
    """

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_router_num_groups: Optional[int] = None
    """
    Number of groups to divide experts into for group-limited routing.

    When using group-limited routing:
    1. Experts are divided into 'moe_router_num_groups' equal-sized groups
    2. For each token, 'moe_router_group_topk' groups are selected based on sum of
    top-('moe_router_topk'/'moe_router_group_topk') routing scores within each group
    3. From these selected groups, 'moe_router_topk' individual experts are chosen

    Two common use cases:
    - Device-limited routing: Set 'moe_router_num_groups' equal to expert parallel size (EP)
    to limit each token to experts on a subset of devices.
    (See DeepSeek-V2: https://arxiv.org/pdf/2405.04434)
    - Node-limited routing: Set 'moe_router_num_groups' equal to number of nodes in EP group
    to limit each token to experts on a subset of nodes.
    (See DeepSeek-V3: https://arxiv.org/pdf/2412.19437)
    """

    moe_router_group_topk: Optional[int] = None
    """Number of selected groups for group-limited routing."""

    moe_router_pre_softmax: bool = False
    """
    Enable pre-softmax(pre-sigmoid) routing for MoE,
    which means softmax is before the top-k selection.
    By default, softmax is done after top-k.
    """

    moe_router_topk_scaling_factor: Optional[float] = None
    """
    Scaling factor for routing score in top-k selection, only works when moe_router_pre_softmax enabled.
    Defaults to None, which means no scaling.
    """

    moe_router_dtype: str = "float32"
    """
    Data type for routing and expert output weighted averaging.
    Using fp32 or fp64 can improve stability especially when the number of experts is large (e.g. finegrained-moe).
    None means no changes for dtype.
    """

    moe_router_bias_update_rate: float = 1e-3
    """
    The expert bias is updated based on the number of assigned tokens to each expert
    in a global batch, where the bias is increased for the experts with less assigned tokens
    and decreased for the experts with more assigned tokens.
    The default value 1e-3 is same as that used in DeepSeekV3.
    """

    moe_grouped_gemm: bool = False
    """
    When there are multiple experts per rank, compress multiple local (potentially small) gemms
    in a single kernel launch to improve the utilization and performance by leveraging the Grouped
    GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).
    """

    moe_aux_loss_coeff: float = 0.  # 1e-2 would be a good start value for load balance loss.
    """Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended."""

    moe_z_loss_coeff: Optional[float] = None  # 1e-3 would be a good start value for z-loss
    """Scaling coefficient for the z-loss. A starting value of 1e-3 is recommended."""

    moe_input_jitter_eps: Optional[float] = None
    """Add noise to the input tensor by applying jitter with a specified epsilon value."""

    group_wise_a2a: bool = False
    """
    Whether to enable group-wise alltoall communication,
    which can reduce communication time by converting part of intercommunication into intra communication.

    This parameter is effective only when model parallel > 1 and data_parallel equal to expert parallel.
    """

    moe_token_dispatcher_type: str = "alltoall"
    """The type of token dispatcher to use. The default is 'alltoall'.
    Options are 'alltoall', 'alltoall_deredundency' and 'alltoall_zero_redundancy'."""

    moe_enable_deepep: bool = False
    """[Experimental] Enable DeepEP for efficient token dispatching and combine in MoE models."""

    moe_per_layer_logging: bool = False
    """Enable per-layer logging for MoE, currently supports auxiliary loss and z loss."""

    moe_expert_capacity_factor: Optional[float] = None
    """
    The capacity factor for each expert, None means no token will be dropped.
    The default is None.
    """

    moe_pad_expert_input_to_capacity: bool = False
    """
    If True, pads the input for each expert to match the expert capacity length,
    effective only after the moe_expert_capacity_factor is set.
    The default setting is False.
    """

    moe_token_drop_policy: str = 'probs'
    """
    The policy to drop tokens. Can be either "probs" or "position".
    If "probs", the tokens with the lowest probabilities will be dropped.
    If "position", tokens at the end of each batch will be dropped.
    """

    moe_permute_fusion: bool = False
    """Fuse token rearrangement ops during token dispatching."""

    moe_apply_probs_on_input: bool = False
    """Apply probs on input of experts instead of applying after activation and glu."""

    # MindFormers New
    shared_expert_num: int = 0
    """Number of shared experts."""

    ##################
    # Context Parallel
    ##################

    cp_comm_type: Optional[Union[str, List[str]]] = None
    """
    Reserved interface, will be supported in subsequent versions.

    Inter-NPU communication type for context parallelism.
    - str: all layers share same communication type.
    - List[str]: each layer has its separate communication type.

    cp_comm_type of each layer can be "p2p" or "all_gather" or "a2a" or "a2a+p2p".
    - "p2p": Exchange KV chunks with P2P communications in ring topology. P2P is async and can be
        overlapped with attention compute.
    - "all_gather": All-gather to get full sequence of KV before attention. The all-gather is not
        async, and cannot be overlapped.
    - "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP group, and gather to get
        full sequence of QKV.
    - "a2a+p2p": A hierarchical implementation of context parallelism to attention.

    It uses A2A communications in low-level CP groups, and P2P communications in high-level CP groups.
    """

    def __post_init__(self):
        """
        Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        super().__post_init__()

        self.compute_dtype = convert_str_to_mstype(self.compute_dtype)
        self.layernorm_compute_dtype = convert_str_to_mstype(self.layernorm_compute_dtype)
        self.rotary_dtype = convert_str_to_mstype(self.rotary_dtype)
        self.moe_router_dtype = convert_str_to_mstype(self.moe_router_dtype)
        self.softmax_compute_dtype = convert_str_to_mstype(self.softmax_compute_dtype)

        if not isinstance(self.hidden_dropout, float) or not 0 <= self.hidden_dropout < 1:
            raise ValueError(f"hidden_dropout should be a float within [0, 1), but get {self.hidden_dropout}.")
        if not isinstance(self.attention_dropout, float) or not 0 <= self.attention_dropout < 1:
            raise ValueError(f"attention_dropout should be a float within [0, 1), but get {self.attention_dropout}.")

        if self.pad_token_id is None:
            self.pad_token_id = 0

        self.mtp_num_layers = self.mtp_num_layers or 0
        if self.mtp_num_layers is not None:
            if self.mtp_num_layers < 0 or not isinstance(self.mtp_num_layers, int):
                raise ValueError(
                    f"mtp_num_layers should be `None` or non-negative integer, but get {self.mtp_num_layers}."
                )
            elif self.mtp_num_layers > 1:
                raise ValueError(
                    f"The current version only supports the scenario where `mtp_num_layers` = `1` is configured. "
                    f"But get {self.mtp_num_layers}."
                )

        if self.num_attention_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        if self.context_parallel_size > 1 and not self.use_flash_attention:
            raise ValueError("context_parallel is only available for flash attention for now, "
                             "please set use_flash_attention=True.")

        if self.use_flash_attention:
            if self.use_eod_attn_mask_compression and not self.use_ring_attention:
                self.input_layout = "TND"
                if self.attention_dropout != 0:
                    logger.warning("When use TND layout of flash attention, attention_dropout is ignored. Set to 0.")
                    self.attention_dropout = 0.
            elif self.context_parallel_size > 1:
                self.input_layout = "BSH"
            else:
                self.input_layout = "BNSD"

            if self.use_eod_attn_mask_compression and not self.use_ring_attention:
                self.sparse_mode = 3
            elif self.use_attn_mask_compression and not self.use_ring_attention:
                self.sparse_mode = 2
            else:
                self.sparse_mode = 0
        else:
            if self.use_eod_attn_mask_compression or self.use_attn_mask_compression:
                raise ValueError("When use mask compression, use_flash_attention must be True.")
            if self.use_ring_attention:
                raise ValueError("When use ring attention, use_flash_attention must be True.")

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:
            raise ValueError('num_moe_experts must be non None to use expert-parallel.')

        if self.moe_token_dispatcher_type == "alltoall_deredundency" and \
            (self.expert_model_parallel_size < self.npu_nums_per_device):
            raise ValueError(
                f"expert_model_parallel_size must be greater than or equal to npu_nums_per_device when using "
                f"'alltoall_deredundency', but got expert_model_parallel_size={self.expert_model_parallel_size} "
                f"< npu_nums_per_device={self.npu_nums_per_device}."
            )

        if self.num_moe_experts is not None and self.num_moe_experts <= 0:
            raise ValueError('num_moe_experts must be non-negative.')

        if self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size

        if self.moe_shared_expert_intermediate_size is not None:
            if self.shared_expert_num == 0:
                logger.warning("The hidden-size of shared experts ('moe_shared_expert_intermediate_size') is set, "
                               "but get shared_expert_num = 0. The shared_expert_num will be ignored.")
            elif self.moe_shared_expert_intermediate_size != self.moe_ffn_hidden_size * self.shared_expert_num:
                logger.warning(
                    f'moe_shared_expert_intermediate_size should be '
                    f'num_shared_experts ({self.shared_expert_num}) * '
                    f'ffn_size_of_each_shared_expert ({self.moe_ffn_hidden_size}), '
                    f'but got {self.moe_shared_expert_intermediate_size}. '
                    f'moe_shared_expert_intermediate_size ({self.moe_shared_expert_intermediate_size}) will be applied.'
                )
        elif self.shared_expert_num > 0:
            self.moe_shared_expert_intermediate_size = self.moe_ffn_hidden_size * self.shared_expert_num

        if self.moe_expert_capacity_factor is not None:
            if self.moe_expert_capacity_factor < 0:
                self.moe_expert_capacity_factor = None
            if self.moe_router_load_balancing_type not in ["sub_seq_aux_loss", "seq_aux_loss", "gbs_aux_loss"]:
                raise ValueError(
                    'moe_expert_capacity_factor only works with supported load balancing types: '
                    'sub_seq_aux_loss, seq_aux_loss, gbs_aux_loss'
                )

        if self.moe_pad_expert_input_to_capacity:
            if self.moe_expert_capacity_factor is None:
                raise ValueError(
                    'moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity'
                )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.apply_rope_fusion:
            if self.multi_latent_attention:
                raise ValueError("multi_latent_attention does not support apply_rope_fusion.")

        if self.multi_latent_attention and self.rotary_interleaved:
            raise ValueError("rotary_interleaved does not work with multi_latent_attention.")

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std, self.params_dtype)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std,
                self.num_layers,
                self.params_dtype
            )

        if self.num_moe_experts is not None:
            assert not self.add_bias_linear, "Bias is not supported for MoE"

        if self.moe_router_enable_expert_bias and self.moe_router_score_function != "sigmoid":
            raise ValueError(
                "Expert bias for aux-loss-free routing only supports sigmoid score function."
                "Please set --moe-router-score-function sigmoid for sigmoid score function."
            )

        if (
                self.moe_router_topk == 1
                and self.moe_router_score_function == 'softmax'
                and not self.moe_router_pre_softmax
                and self.moe_router_load_balancing_type != 'sinkhorn'
        ):
            # Requires applying softmax before selecting the top-k when k is 1,
            # since softmax on a [num_tokens, 1] would yield a zero gradient.
            raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")

        if self.moe_router_group_topk:
            if not self.moe_router_num_groups:
                raise ValueError(
                    "When using group limited routing, moe_router_num_groups must be specified."
                )
            else:
                assert self.num_moe_experts % self.moe_router_num_groups == 0, (
                    f"num_moe_experts ({self.num_moe_experts}) should be divisible by "
                    f"moe_router_num_groups ({self.moe_router_num_groups})."
                )
                assert self.moe_router_group_topk <= self.moe_router_num_groups, (
                    f"moe_router_group_topk ({self.moe_router_group_topk}) should be smaller than "
                    f"moe_router_num_groups ({self.moe_router_num_groups})."
                )
                assert self.moe_router_topk % self.moe_router_group_topk == 0, (
                    f"`moe_router_topk` must be divisible by `moe_router_group_topk`. "
                    f"Got moe_router_topk={self.moe_router_topk} and "
                    f"moe_router_group_topk={self.moe_router_group_topk}."
                )

        if (
                self.num_moe_experts is not None
                and self.num_moe_experts >= 32
                and not self.moe_router_dtype
        ):
            logger.warning(
                "Using a large number of experts (e.g. >=32) without fp32 routing. "
                "Consider enabling moe_router_dtype for better numerical stability."
            )

        if self.first_k_dense_replace:
            moe_layer_freq_template = [0] * self.first_k_dense_replace + [1] * (
                self.num_layers - self.first_k_dense_replace)
            if isinstance(self.moe_layer_freq, int) and not isinstance(self.moe_layer_freq, bool):
                if self.moe_layer_freq > 1:
                    raise ValueError(
                        "Configuration conflict: 'first_k_dense_replace' cannot be "
                        "used together with 'moe_layer_freq > 1'."
                    )
                else:
                    self.moe_layer_freq = moe_layer_freq_template
            elif isinstance(self.moe_layer_freq, list):
                if self.moe_layer_freq != moe_layer_freq_template:
                    raise ValueError(
                        f"'moe_layer_freq' should be {moe_layer_freq_template}, "
                        f"but got {self.moe_layer_freq}"
                    )
            else:
                raise TypeError("'moe_layer_freq' should be <int> or <list[int]>, "
                                f"but got {type(self.moe_layer_freq)}")
            if self.first_k_dense_replace > self.num_layers:
                raise ValueError(
                    f"'first_k_dense_replace'({self.first_k_dense_replace}) should not be bigger "
                    f"than 'num_layers'({self.num_layers})."
                )
        elif self.moe_layer_freq != 1 or isinstance(self.moe_layer_freq, bool):
            if isinstance(self.moe_layer_freq, int) and not isinstance(self.moe_layer_freq, bool):
                if self.moe_layer_freq > self.num_layers:
                    raise ValueError(
                        f"'moe_layer_freq'({self.moe_layer_freq}) should not be bigger "
                        f"than 'num_layers'({self.num_layers})."
                    )
            elif isinstance(self.moe_layer_freq, list):
                if len(self.moe_layer_freq) != self.num_layers:
                    raise ValueError(
                        f"Length of 'moe_layer_freq'({self.moe_layer_freq}) "
                        f"must be equal to 'num_layers'({self.num_layers})."
                    )
                for num in self.moe_layer_freq:
                    if num not in (0, 1):
                        raise ValueError("Invalid 'moe_layer_freq', "
                                         f"numbers in 'moe_layer_freq'({self.moe_layer_freq}) must be equal to 1 or 0")
            else:
                raise TypeError("'moe_layer_freq' should be <int> or <list[int]>, "
                                f"but got {type(self.moe_layer_freq)}")

        self.is_dryrun = os.environ.get('MS_SIMULATION_LEVEL', '0') != '0'
        if self.is_dryrun:
            if self.num_moe_experts is not None and self.seq_length % self.num_moe_experts != 0:
                raise ValueError(
                    f"When using moe_dry_run, seq_length ({self.seq_length}) must be divisible by "
                    f"num_moe_experts ({self.num_moe_experts})"
                    )
            elif self.moe_token_dispatcher_type != "alltoall":
                raise ValueError(
                    "When using moe_dry_run, moe_token_dispatcher_type must be alltoall."
                    )

        if isinstance(self.rope_scaling, dict):
            self.position_embedding_type = (self.rope_scaling.pop("type", None) or
                                            self.rope_scaling.pop("rope_type", None))
            self.rotary_scaling_factor = self.rope_scaling.pop("factor")
            self.max_position_embeddings = self.rope_scaling.pop("original_max_position_embeddings",
                                                                 None) or self.seq_length
            for k, v in self.rope_scaling.items():
                setattr(self, k, v)
            del self.rope_scaling

        if self.bias_swiglu_fusion and self.hidden_act != 'swiglu':
            raise ValueError(
                "When using bias_swiglu_fusion, hidden_act must be swiglu."
            )

        if (self.moe_router_load_balancing_type is not None
                and not isinstance(self.moe_router_load_balancing_type, str)):
            raise TypeError("moe_router_load_balancing_type must be a string, "
                            f"but got {type(self.moe_router_load_balancing_type)}.")

        if self.moe_aux_loss_coeff is not None and not isinstance(self.moe_aux_loss_coeff, (float, int)):
            raise TypeError(f"moe_aux_loss_coeff must be a float or int, but got {type(self.moe_aux_loss_coeff)}.")

        if ms.get_auto_parallel_context("pipeline_scheduler") == "zero_bubble_v":
            if self.virtual_pipeline_model_parallel_size != 2:
                raise ValueError(
                    f"When zero_bubble_v is enabled, pp_interleave_num must be set to 2. "
                    f"But get {self.virtual_pipeline_model_parallel_size}.")
            if self.pipeline_model_parallel_size < 2:
                raise ValueError(
                    f"When zero_bubble_v is enabled, pp must be greater than or equal to 2. "
                    f"But get {self.pipeline_model_parallel_size}.")
            if self.micro_batch_num < 2 * self.pipeline_model_parallel_size:
                raise ValueError(
                    f"When zero_bubble_v is enabled, micro_batch_num({self.micro_batch_num}) >= 2 * stage_num"
                    f"({self.pipeline_model_parallel_size}) must be met.")
            if isinstance(self.recompute, (list, tuple)):
                if all(isinstance(item, (int, bool)) for item in self.recompute) or len(self.recompute) < 2:
                    raise ValueError(
                        "When zero_bubble_v is enabled, "
                        "'recompute' must provide explicit 2D configuration for each interleave, "
                        "such as [[stage0_recompute, stage1_recompute], [stage0_recompute, stage1_recompute]].")

        if self.recompute_slice_activation:
            raise ValueError("For recompute, `recompute_slice_activation` is not supported in Mcore.")

        self._validate_param_init_std_rules()

    def _validate_param_init_std_rules(self):
        """Validate and compile decoupling initialization rules."""
        rules = self.param_init_std_rules

        # Only process if rules are provided (non-empty list)
        if rules:
            # Ensure rules is a list
            if not isinstance(rules, list):
                raise TypeError(
                    f"param_init_std_rules must be a list, "
                    f"but got {type(rules)}(value: {rules})"
                )

            for idx, rule in enumerate(rules):
                if not isinstance(rule, dict):
                    raise TypeError(f"Rule {idx} is not a dict, got {type(rule)}")

                target = rule.get("target")
                init_std = rule.get("init_method_std")

                # Validate 'target' field
                if not isinstance(target, str):
                    raise TypeError(f"Rule {idx}: 'target' must be a string, but got {type(target)}")

                # Validate 'init_method_std' field
                if not isinstance(init_std, (int, float)):
                    raise TypeError(
                        f"Rule {idx}: 'init_method_std' must be a number, "
                        f"but got {type(init_std)}"
                    )
                if init_std < 0:
                    raise ValueError(f"Rule {idx}: 'init_method_std' must be >= 0, but got {init_std}")

                # Compile the regex pattern and replace the string in-place
                compiled_pattern = re.compile(target)
                rule["target"] = compiled_pattern


@dataclass
class MLATransformerConfig(TransformerConfig):
    """
    Configuration object for MindSpore Transformer's Multi-Latent Attention (MLA) transformers.

    The initialization function has an argument for each parameter, including those in ModelParallelConfig.
    Included YaRN RoPE parameters that is fused in MLA.
    """

    multi_latent_attention: bool = True
    """Whether to use Multi-Latent Attention."""

    q_lora_rank: int = 512
    """Rank of Query tensor's low rank representation."""

    kv_lora_rank: int = 512
    """Rank of Key and Value tensors' low rank representation."""

    qk_head_dim: int = 128
    """Dimension of the head in the QK projection. q_head_dim = qk_head_dim + qk_pos_emb_head_dim"""

    qk_pos_emb_head_dim: int = 64
    """Dimension of the position embedding in the QK projection."""

    v_head_dim: int = 128
    """Dimension of the head in the V projection."""

    normalization: str = "RMSNorm"
    """Default normalization layer for MLA models is RMSNorm."""

    rope_type: str = "yarn"
    """Type of RoPE to use. Default to yarn, options are rope and yarn."""

    rotary_percent: float = 1.0
    """Rotary percent for the rotary embeddings, used by rope."""

    rotary_scaling_factor: float = 40.0
    """Rotary scaling factor for the rotary embeddings, used by yarn."""

    max_position_embeddings: int = 4096
    """Maximum position embeddings for the original model, used by yarn."""

    beta_fast: float = 32.0
    """Beta fast for YaRN RoPE, used by yarn."""

    beta_slow: float = 1.0
    """Beta slow for YaRN RoPE, used by yarn."""

    mscale: float = 0.707
    """Mscale for YaRN RoPE in Multi-Latent Attention, used by yarn."""

    mscale_all_dim: float = 0.707
    """Mscale all dimensions for YaRN RoPE in Multi-Latent Attention, used by yarn."""

default_transformer_config = TransformerConfig(num_attention_heads=1, num_layers=1)
