# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Modified some config parameters to adapt to MindSpore Transformer.
"""Model Parallel Config"""

from dataclasses import dataclass
from typing import Optional, Union

from mindformers.parallel_core.mf_model_config import convert_str_to_mstype


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
    (https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/configuration.html)
    """

    vocab_emb_dp: Optional[bool] = False
    """
    Whether to split the vocabulary only along the dp dimension.
    This setting is not supported to be configured as True at present; 
    otherwise, it will be converted to False automatically.
    """

    ###################
    # Training
    ###################

    params_dtype: str = "float32"
    """dtype used when initializing the weights."""

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
            raise ValueError("Can not use sequence parallelism without tensor parallelism")
