# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Modified some config parameters to adapt to MindSpore Transformer.
"""Model Parallel Config"""

from dataclasses import dataclass
from typing import Callable, Optional, Union

from mindformers.parallel_core.mf_model_config import convert_str_to_mstype
from mindformers.tools.logger import logger


@dataclass
class ModelParallelConfig:
    """
    Base configuration for MindSpore Transformer's Core

    The initialization function has an argument for each parameter.
    """

    ###################
    # Model parallelism
    ###################

    data_parallel_size: int = 1
    """
    Data parallelism. The training data is partitioned into multiple micro-batches,
    with each batch assigned to a distinct device for distributed parallel processing.
    Each accelerator independently processes different batches in parallel,
    and the gradients or outputs are subsequently synchronized and aggregated across devices.
    """

    tensor_model_parallel_size: int = 1
    """Intra-layer model parallelism. Splits tensors across NPU ranks."""

    pipeline_model_parallel_size: int = 1
    """Inter-layer model parallelism. Splits transformer layers across NPU ranks."""

    virtual_pipeline_model_parallel_size: Optional[int] = None
    """
    Interleaved pipeline parallelism is used to improve performance by reducing the pipeline bubble.
    Considers a transformer block as a list of smaller transformer (virtual) blocks.
    The number of virtual blocks per pipeline model parallel rank is the virtual model parallel size.
    """

    sequence_parallel: bool = False
    """
    Makes tensor parallelism more memory efficient for LLMs (20B+) by parallelizing layer norms
    and dropout sequentially.
    See Reducing Activation Recomputation in Large Transformer Models
    (https://arxiv.org/abs/2205.05198) for more details.
    """

    context_parallel_size: int = 1
    """Splits network input along sequence dimension across NPU ranks."""

    hierarchical_context_parallel_sizes: int = 1
    """
    Reserved interface.

    Degrees of the hierarchical context parallelism.
    Users should provide a list to specify the sizes for different levels.
    Taking the a2a+p2p cp comm type as example, it contains groups of two levels,
    so the first value of the list indicates the group size of the a2a communication type,
    and the second value indicates the group size of the p2p communication type.
    """

    expert_model_parallel_size: int = 1
    """Distributes Moe Experts across sub data parallel dimension."""

    expert_tensor_parallel_size: Optional[int] = None
    """Intra-layer tensor model parallelism for expert layer. Splits tensors across NPU ranks."""

    # Mindformers New
    micro_batch_num: Optional[int] = 1
    """MicroBatch size for Pipeline Parallel. Default: 1."""

    seq_split_num: Optional[int] = 1
    """Sequence split number in sequence pipeline parallel mode. Default: 1."""

    gradient_aggregation_group: int = 4
    """The fusion group size of the optimizer state sharding. Default: 4."""

    offset: Optional[Union[int, list]] = 0
    """Offset of transformer layer when set pipeline stage number. Default: 0."""

    ulysses_degree_in_cp: int = 1
    """
    The number of parallel slices of the Ulysses sequence.
    For configuration method of distributed parallel parameters,
    refer to the contents of the Parallel Configuration section
    in MindSpore Transformers configuration description:
    (https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/configuration.html)
    """

    vocab_emb_dp: Optional[bool] = True
    """Whether to split the vocabulary only along the dp dimension. Default: True."""

    ###################
    # Training
    ###################

    fp16: bool = False
    """If true, train with fp16 mixed precision training."""

    bf16: bool = False
    """If true, train with bf16 mixed precision training."""

    params_dtype: str = "float32"
    """dtype used when initializing the weights."""

    finalize_model_grads_func: Optional[Callable] = None
    """
    Function that finalizes gradients on all workers.
    Could include ensuring that grads are all-reduced across data parallelism, pipeline parallelism,
    and sequence parallelism dimensions.
    """

    grad_scale_func: Optional[Callable] = None
    """
    If using loss scaling, this function should take the loss and return the scaled loss.
    If None, no function is called on the loss.
    """

    grad_sync_func: Optional[Callable] = None
    """
    Function that launches asynchronous gradient reductions (e.g. distributed optimizer gradient reduce-scatters).
    The function should take one argument: an iterable of parameters whose gradients are to be synchronized.
    """

    param_sync_func: Optional[Callable] = None
    """
    Function that launches asynchronous parameter synchronizations (e.g. distributed optimizer parameter all-gathers).
    The function should take one argument: an iterable of parameters to be synchronized.
    """

    num_microbatches_with_partial_activation_checkpoints: Optional[int] = None
    """
    If int, set the number of microbatches where not all of the layers will be checkpointed and recomputed.
    The rest of the microbatches within the window of maximum outstanding
    microbatches will recompute all layers (either full recompute or selective recompute).
    If None, the checkpoint and recompute will be left up to the forward_step function.
    """

    ###################
    # CPU Offloading
    ###################
    cpu_offloading: bool = False
    """Enable offload of the transformer block or not. Default: False."""

    # MindFormers New
    cpu_offloading_num_layers: Optional[Union[list, dict]] = None
    """
    Configuration for layer swapping.
    Each item in the list specifies the `backward_prefetch` value for a specific layer.
    Default: None.
    """

    op_swap: Optional[Union[list, dict]] = None
    """
    Configuration for operator swapping.
    Each item in the list specifies the `backward_prefetch` value for operators matching a specific pattern.
    Default: None.
    """

    default_prefetch: int = 1
    """
    Number of operators to prefetch activations before the backward FlashAttention (FA) operator.
    In the context of static graph execution, since the activation values that have been offloaded
    need to be retrieved again during the backward pass, and retrieving data from CPU back to NPU incurs latency.
    """

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        self.params_dtype = convert_str_to_mstype(self.params_dtype)

        if self.sequence_parallel and self.tensor_model_parallel_size <= 1:
            logger.warning(
                "Sequence parallelism can't be enabled without tensor parallelism. Set to False."
            )
            self.sequence_parallel = False

default_dpmp_config = ModelParallelConfig()
