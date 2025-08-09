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
"""Linear units for tensor parallelism"""

__all__ = [
    "ColumnParallelLinear",
    "QKVParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "ReplicatedLinear"
]

from typing import Callable, List, Optional
from abc import abstractmethod

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor, mint, nn, ops

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.tensor_parallel.mappings import (gather_from_model_parallel_region,
                                                                          reduce_from_model_parallel_region,
                                                                          scatter_to_model_parallel_region)
from mindformers.parallel_core.inference.utils import divide

from mindformers.parallel_core.inference.parallel_state import ProcessGroup, default_pgs
from mindformers.parallel_core.inference.weights_utils import (set_weight_attrs, split_loaded_weight,
                                                               deal_linear_q_up_weight, deal_linear_kv_up_weight,
                                                               deal_linear_kv_down_weight)
from mindformers.parallel_core.inference.tensor_parallel.quantization.base_config import QuantizeMethodBase
from mindformers.version_control import is_310p
from mindformers.models.utils import format_type


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(self, layer: ms.nn.Cell, input_size_per_partition: int,
                       output_partition_sizes: List[int], params_dtype, **extra_weight_attrs):
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            params_dtype: Datatype of the parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self,
              layer: ms.nn.Cell,
              x: ms.Tensor,
              weight: Tensor,
              bias: Optional[ms.Tensor] = None) -> ms.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(self, layer: ms.nn.Cell, input_size_per_partition: int,
                       output_partition_sizes: List[int], params_dtype, **extra_weight_attrs):
        if extra_weight_attrs.get('transpose_b'):
            weight = Parameter(
                mint.zeros(
                    (int(sum(output_partition_sizes)),
                     int(input_size_per_partition)),
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
        else:
            weight = Parameter(
                mint.zeros(
                    (int(input_size_per_partition),
                     int(sum(output_partition_sizes))),
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
        self.input_size_per_partition = int(input_size_per_partition)
        self.output_size_per_partition = int(sum(output_partition_sizes))
        if extra_weight_attrs.get('transpose_b'):
            set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        else:
            set_weight_attrs(weight, {"input_dim": 0, "output_dim": 1})
        # layer.register_parameter("weight", weight)
        layer.insert_param_to_cell("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        self.matmul = ops.MatMul(transpose_b=extra_weight_attrs.get('transpose_b'))
        self.cast = ops.Cast()

    def apply(self, layer: ms.nn.Cell, x: Tensor, weight: Tensor, bias: Parameter = None):
        origin_dtype = x.dtype
        output_shape = x.shape[:-1] + (self.output_size_per_partition,)

        x = mint.reshape(x, (-1, self.input_size_per_partition))
        x = self.cast(x, layer.compute_dtype)
        weight = self.cast(weight, layer.compute_dtype)
        output_parallel = self.matmul(x, weight)

        if bias is not None and not layer.skip_bias_add:
            bias = self.cast(bias, layer.compute_dtype)
            output_parallel = mint.add(output_parallel, bias)

        output_parallel = mint.reshape(output_parallel, output_shape)
        output_parallel = self.cast(output_parallel, origin_dtype)
        return output_parallel


class LinearBase(ms.nn.Cell):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            skip_bias_add: bool = False,
            params_dtype: mstype = mstype.float32,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        # Currently does not support quantization, only use UnquantizedLinearMethod.
        self.quant_method: Optional[
            QuantizeMethodBase] = UnquantizedLinearMethod()
        self.param_load_counts: Dict[str, int] = {}

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        raise NotImplementedError

    def format_to_nz(self, param, merge_count=1):
        current_count = self.param_load_counts.get(param.name, 0) + 1
        self.param_load_counts[param.name] = current_count

        if current_count == merge_count:
            cast_weight = ops.auto_generate.format_cast(param, format_type['nz'])
            param.set_data(cast_weight)
            del self.param_load_counts[param.name]

class ColumnParallelLinear(LinearBase):
    r"""
    The dense layer with weight sliced on second dimension by tensor parallel size.
    This layer implements the operation as:

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

    where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
    and :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True).

    Args:
        input_size (int): The number of channels in the input space.
        output_size (int): The number of channels in the output space.
        config (dict): Transformer configuration.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable init_method parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: False.
        skip_bias_add: This was added to enable performance optimizations where bias can be fused with other
                       element-wise operations. We skip adding bias but instead return it. Default: False.
        skip_weight_param_allocation (bool): Specifies whether skip the initialization of weight parameter.
            When set True, an weight tensor should be passed to construct function. Default: False.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
        compute_dtype (dtype.Number): The computation type. Default: mstype.bfloat16.
        tp_group (ProcessGroup): The process_group this linear layer used. Default: default_pgs.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `input_size` in `Args` should be equal
          to :math:`in\_channels` in `Inputs`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Raises:
        ValueError: `skip_weight_param_allocation=True` but weight_tensor is not passed to construct function.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            config: TransformerConfig,
            init_method: Callable = "normal",
            bias: bool = True,
            gather_output: bool = False,
            stride: int = 1,
            keep_master_weight_for_test: bool = False,
            skip_bias_add: bool = False,
            skip_weight_param_allocation: bool = False,
            embedding_activation_buffer: Optional[List[Tensor]] = None,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
            transpose_b: bool = True,
            compute_dtype: mstype = mstype.bfloat16,
            tp_group: ProcessGroup = default_pgs,
    ):
        super(ColumnParallelLinear, self).__init__(input_size, output_size, skip_bias_add, config.params_dtype)
        if stride > 1:
            raise NotImplementedError("For ColumnParallelLinear, `stride > 1` is not supported for now, "
                                      "but got `stride={}`".format(stride))
        if keep_master_weight_for_test:
            raise NotImplementedError(
                "For ColumnParallelLinear, `keep_master_weight_for_test` is not supported for now")
        if skip_bias_add:
            raise NotImplementedError("For ColumnParallelLinear, `skip_bias_add=True` is not supported for now")
        if embedding_activation_buffer is not None:
            raise NotImplementedError(
                "For ColumnParallelLinear, `embedding_activation_buffer` is not supported for now")
        if is_expert:
            raise NotImplementedError("For ColumnParallelLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For ColumnParallelLinear, `tp_comm_buffer_name` is not supported for now")

        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.init_method = init_method
        self.has_bias = bias
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.transpose_b = transpose_b
        self.compute_dtype = compute_dtype
        self.params_dtype = config.params_dtype
        self.cast = P.Cast()
        self.matmul = P.MatMul(transpose_b=self.transpose_b)

        self.tp_group = tp_group
        self.tensor_parallel_group_size = self.tp_group.size
        self.output_size_per_partition = divide(output_size, self.tensor_parallel_group_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tensor_parallel_group_size)
                for output_size in self.output_sizes
            ]

        bias_shape = (self.output_size_per_partition,)
        if self.has_bias:
            self.bias = Parameter(
                mint.empty(bias_shape, dtype=self.params_dtype),
                name="bias"
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.bias = None

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
            transpose_b=self.transpose_b
        )

    def construct(self, input_, weight=None):
        """
        Forward of ColumnParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "For ColumnParallelLinear, weight was not supplied to construct(), "
                    "and `skip_weight_param_allocation` is True."
                    )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape.
            experted_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != experted_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {experted_shape} as expected."
                )
        output_parallel = self.quant_method.apply(self, input_, weight, self.bias)

        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel, self.tp_group)
        else:
            output = output_parallel
        return output

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (self.tensor_parallel_group_size, 1) if self.transpose_b else (1, self.tensor_parallel_group_size)

        state_dict = {}
        if not self.skip_weight_param_allocation:
            state_dict[self.weight.name] = {'shape': self.weight.shape,
                                            'shard': w_shard}
        if self.bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (self.tensor_parallel_group_size,)}
        return state_dict

    def weight_loader(self, param, loaded_weight, loaded_shard_id: Optional[str] = None):
        """
        Load and process weights for ColumnParallelLinear layer with support for sharded loading.

        This method handles the loading of weights that have been partitioned along the output dimension
        according to tensor parallelism.

        Args:
            param: The parameter tensor to load weights into.
            loaded_weight: The weight tensor loaded from checkpoint.
            loaded_shard_id: Optional identifier for sharded weight loading.

       """
        tp_rank = self.tp_group.rank
        shard_dim = getattr(param, "output_dim", None)
        shard_size = self.output_size_per_partition
        loaded_weight = loaded_weight[:]
        if loaded_shard_id is not None:
            if loaded_shard_id == 'q_up':
                loaded_weight = deal_linear_q_up_weight(loaded_weight, self.config, shard_dim, shard_size=shard_size)
            if loaded_shard_id == 'kv_up':
                loaded_weight = deal_linear_kv_up_weight(loaded_weight, self.config, shard_dim, shard_size=shard_size)
        else:
            start_idx = tp_rank * shard_size
            loaded_weight = split_loaded_weight(loaded_weight, shard_dim, start_idx, shard_size)

        if loaded_weight.shape == ():
            loaded_weight = loaded_weight.reshape(1)

        if param.shape != loaded_weight.shape:
            raise ValueError(
                f"'{param.name}.shape' should be equal to 'loaded_weight.shape',"
                f" but got the shape of param is {param.shape} and the shape of weight is{loaded_weight.shape}")
        param.set_data(ms.from_numpy(loaded_weight))
        if is_310p() and param.name.endswith("weight"):
            self.format_to_nz(param)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's FFN transformation.

    Linear layers for the linear transformation of the gate, linear_fc1
    vectors in the mlp layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.

    Args:
        hidden_size: input hidden state size of the transformer.
        config (dict): Transformer configuration.
        bias: If true, add bias.2
        gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: False.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
        compute_dtype (dtype.Number): The computation type. Default: mstype.bfloat16.
        tp_group (ProcessGroup): The process_group this linear layer used. Default: default_pgs.
    """

    def __init__(self,
                 hidden_size: int,
                 ffn_hidden_size: int,
                 *,
                 config: TransformerConfig,
                 bias: bool = True,
                 gather_output: bool = False,
                 is_expert: bool = False,
                 transpose_b: bool = True,
                 compute_dtype: mstype = None,
                 tp_group: ProcessGroup = default_pgs,
                 ):
        self.params_dtype = config.params_dtype

        # Divide the weight matrix along the last dimension.
        self.tp = tp_group
        output_size = (
            ffn_hidden_size
        )
        self.output_sizes = [
            ffn_hidden_size,
            ffn_hidden_size,
        ]
        super().__init__(
            input_size=hidden_size,
            output_size=output_size,
            config=config,
            bias=bias,
            gather_output=gather_output,
            is_expert=is_expert,
            transpose_b=transpose_b,
            compute_dtype=compute_dtype,
            tp_group=tp_group,
        )

    def weight_loader(self,
                      param,
                      loaded_weight,
                      loaded_shard_id: Optional[str] = None):
        output_dim = getattr(param, "output_dim", None)
        tp_rank = self.tp_group.rank
        tp_size = self.tp_group.size
        shard_size = 0
        shard_offset = 0
        if loaded_shard_id is not None:
            if loaded_shard_id == 'gating':
                array_id = 0
            elif loaded_shard_id == 'hidden':
                array_id = 1
            shard_offset = sum(self.output_sizes[:array_id]) // tp_size
            shard_size = self.output_sizes[array_id] // tp_size

        start_idx = tp_rank * shard_size
        loaded_weight = split_loaded_weight(loaded_weight, output_dim,
                                            start_idx, shard_size)

        if loaded_weight.shape == (shard_size, param.shape[1]):
            param[shard_offset:shard_offset + shard_size, :] = ms.from_numpy(loaded_weight)
        else:
            raise ValueError(
                f"'{param.name}.shape' should be equal to 'loaded_weight.shape',"
                f" but got the shape of param is {(shard_size, param.shape[1])} and "
                f"the shape of weight is{loaded_weight.shape}")
        if is_310p() and param.name.endswith("weight"):
            self.format_to_nz(param, 2)


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        config (dict): Transformer configuration.
        bias: If true, add bias.
        gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: False.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
        compute_dtype (dtype.Number): The computation type. Default: mstype.bfloat16.
        tp_group (ProcessGroup): The process_group this linear layer used. Default: default_pgs.
    """

    def __init__(
            self,
            hidden_size: int,
            head_size: int,
            total_num_heads: int,
            total_num_kv_heads: int,
            *,
            config: TransformerConfig,
            bias: bool = True,
            gather_output: bool = False,
            transpose_b: bool = True,
            compute_dtype: mstype = None,
            tp_group: ProcessGroup = default_pgs,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.params_dtype = config.params_dtype

        # Divide the weight matrix along the last dimension.
        self.tp = tp_group
        tp_size = self.tp.size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        output_size = (
            (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        )
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]

        super().__init__(
            input_size=hidden_size,
            output_size=output_size,
            config=config,
            bias=bias,
            gather_output=gather_output,
            transpose_b=transpose_b,
            compute_dtype=compute_dtype,
            tp_group=tp_group,
        )

    def weight_loader(self,
                      param,
                      loaded_weight,
                      loaded_shard_id: Optional[str] = None):
        output_dim = getattr(param, "output_dim", None)
        tp_rank = self.tp_group.rank
        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == "k":
            shard_offset = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        elif loaded_shard_id == "v":
            shard_offset = (self.num_heads +
                            self.num_kv_heads) * self.head_size
            shard_size = self.num_kv_heads * self.head_size

        if loaded_shard_id == "q":
            shard_id = tp_rank
        else:
            shard_id = tp_rank // self.num_kv_head_replicas
        start_idx = shard_id * shard_size
        loaded_weight = split_loaded_weight(loaded_weight, output_dim,
                                            start_idx, shard_size)
        loaded_weight = ms.from_numpy(loaded_weight)

        if param.name.endswith("weight"):
            if loaded_weight.shape != (shard_size, param.shape[1]):
                raise ValueError(
                    f"'{param.name}.shape' should be equal to 'loaded_weight.shape',"
                    f" but got the shape of param is {(shard_size, param.shape[1])} and "
                    f"the shape of weight is{loaded_weight.shape}")
        if param.name.endswith("bias"):
            if loaded_weight.shape != (shard_size,):
                raise ValueError(
                    f"'{param.name}.shape' should be equal to 'loaded_weight.shape',"
                    f" but got the shape of param is {(shard_size,)} and "
                    f"the shape of weight is{loaded_weight.shape}")
        param[shard_offset:shard_offset + shard_size] = loaded_weight
        if is_310p() and param.name.endswith("weight"):
            self.format_to_nz(param, 3)


class RowParallelLinear(LinearBase):
    r"""
    The dense layer with weight sliced on first dimension by tensor parallel size.
    This layer implements the operation as:

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

    where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
    and :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True).

    Args:
        input_size (int): The number of channels in the input space.
        output_size (int): The number of channels in the output space.
        config (dict): Transformer configuration.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable init_method parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The values
            of str refer to the function `initializer`. Default: 'zeros'.
        input_is_parallel (bool): Specifies whether the input tensor has already been sliced on last dimension.
        skip_bias_add: This was added to enable performance optimizations where bias can be fused with other
            element-wise operations. We skip adding bias but instead return it. Default: False.
        delay_allreduce (bool): Whether to delay allreduce during forward function. Default: False
        is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
        compute_dtype (dtype.Number): The computation type. Default: mstype.bfloat16.
        tp_group (ProcessGroup): The process_group this linear layer used. Default: default_pgs.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `input_size` in `Args` should be equal
          to :math:`in\_channels` in `Inputs`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            config: TransformerConfig,
            init_method: Callable = "normal",
            bias: bool = True,
            input_is_parallel: bool = True,
            skip_bias_add: bool = False,
            stride: int = 1,
            keep_master_weight_for_test: bool = False,
            delay_allreduce: bool = False,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
            transpose_b: bool = True,
            compute_dtype: mstype = mstype.bfloat16,
            tp_group: ProcessGroup = default_pgs,
    ):
        super(RowParallelLinear, self).__init__(input_size, output_size, skip_bias_add, config.params_dtype)
        if stride > 1:
            raise NotImplementedError("For RowParallelLinear, `stride > 1` is not supported for now, "
                                      "but got `stride={}`".format(stride))
        if skip_bias_add:
            raise NotImplementedError("For RowParallelLinear, `skip_bias_add=True` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError("For RowParallelLinear, `keep_master_weight_for_test=True` "
                                      "is not supported for now.")
        if is_expert:
            raise NotImplementedError("For RowParallelLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name:
            raise NotImplementedError("For RowParallelLinear, `tp_comm_buffer_name` is not supported for now.")
        if delay_allreduce and bias:
            raise RuntimeError(
                "In RowParallelLinear, `delay_allreduce` and `has_bias` cannot be enabled simultaneously, "
                "otherwise the accuracy will be incorrect."
            )

        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.input_is_parallel = input_is_parallel
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add

        self.tp_group = tp_group
        self.tensor_parallel_group_size = self.tp_group.size
        self.input_size_per_partition = divide(input_size, self.tensor_parallel_group_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]
        self.compute_dtype = compute_dtype
        self.delay_allreduce = delay_allreduce
        self.transpose_b = transpose_b
        self.params_dtype = config.params_dtype
        self.cast = P.Cast()
        self.matmul = P.MatMul(transpose_b=self.transpose_b)

        bias_shape = (self.output_size,)
        if self.has_bias:
            self.bias = Parameter(
                mint.empty(bias_shape, dtype=self.params_dtype),
                name="bias"
                )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.bias = None

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
            transpose_b=self.transpose_b
        )


    def construct(self, input_):
        """
        Forward of RowParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_, self.tp_group)
        bias_ = None if self.tp_group.rank > 0 else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, self.weight, bias_)
        if self.delay_allreduce or self.skip_bias_add:
            output = output_parallel
        else:
            output = reduce_from_model_parallel_region(output_parallel, self.tp_group)
        return output

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (1, self.tensor_parallel_group_size) if self.transpose_b else (self.tensor_parallel_group_size, 1)

        if self.is_expert and self.expert_num > 1:
            w_shard = (1, 1, self.tensor_parallel_group_size) if self.transpose_b \
                else (1, self.tensor_parallel_group_size, 1)

        state_dict = {}
        state_dict[self.weight.name] = {'shape': self.weight.shape,
                                        'shard': w_shard}
        if self.bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (1,)}
        return state_dict

    def weight_loader(self, param, loaded_weight):
        """
        Load and partition weights for RowParallelLinear layer.

        This method handles the loading of weights that have been partitioned along the input dimension
        according to tensor parallelism. Each rank loads its corresponding shard of the weight matrix.

        Args:
            param: The parameter tensor to load weights into.
            loaded_weight: The full weight tensor loaded from checkpoint.

        """
        tp_rank = self.tp_group.rank
        input_dim = getattr(param, "input_dim", None)
        shard_size = self.input_size_per_partition
        start_idx = tp_rank * shard_size
        loaded_weight = split_loaded_weight(loaded_weight, input_dim,
                                            start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == ():
            loaded_weight = loaded_weight.reshape(1)

        if param.shape == loaded_weight.shape:
            param.set_data(ms.from_numpy(loaded_weight))
        else:
            raise ValueError(
                f"'{param.name}.shape' should be equal to 'loaded_weight.shape',"
                f" but got the shape of param is {param.shape} and the shape of weight is{loaded_weight.shape}")
        if is_310p() and param.name.endswith("weight"):
            self.format_to_nz(param)


class ReplicatedLinear(LinearBase):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        config (dict): Transformer configuration.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable init_method parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        bias: If true, add bias.
        gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: False.
        skip_bias_add: This was added to enable performance optimizations where bias can be fused with other
            element-wise operations. We skip adding bias but instead return it.
        skip_weight_param_allocation (bool): Specifies whether skip the initialization of weight parameter.
            When set True, an weight tensor should be passed to construct function. Default: False.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: False.
        transpose_b (bool): Specifies whether the weight parameter will be initialized as a transposed shape.
        compute_dtype (dtype.Number): The computation type. Default: mstype.bfloat16.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            config: TransformerConfig,
            init_method: Callable = "normal",
            bias: bool = True,
            stride: int = 1,
            keep_master_weight_for_test: bool = False,
            skip_bias_add: bool = False,
            skip_weight_param_allocation: bool = False,
            embedding_activation_buffer: Optional[List[Tensor]] = None,
            is_expert: bool = False,
            tp_comm_buffer_name: str = None,
            transpose_b: bool = True,
            compute_dtype: mstype = None
    ):
        super(ReplicatedLinear, self).__init__(input_size, output_size, skip_bias_add, config.params_dtype)
        if stride > 1:
            raise NotImplementedError("For ReplicatedLinear, `stride > 1` is not supported for now, "
                                      "but got `stride={}`".format(stride))
        if skip_bias_add:
            raise NotImplementedError("For ReplicatedLinear, `skip_bias_add=True` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError(
                "For ReplicatedLinear, `keep_master_weight_for_test` is not supported for now")
        if embedding_activation_buffer is not None:
            raise NotImplementedError(
                "For ReplicatedLinear, `embedding_activation_buffer` is not supported for now")
        if is_expert:
            raise NotImplementedError("For ReplicatedLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For ReplicatedLinear, `tp_comm_buffer_name` is not supported for now")

        self.input_size = input_size
        self.output_size = output_size
        self.output_size = [self.output_size]
        self.config = config
        self.init_method = init_method
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.transpose_b = transpose_b
        self.compute_dtype = compute_dtype
        self.params_dtype = config.params_dtype
        self.cast = P.Cast()
        self.matmul = P.MatMul(transpose_b=self.transpose_b)

        self.tensor_parallel_group_size = 1

        bias_shape = (self.output_size,)
        if self.has_bias:
            self.bias = Parameter(
                mint.empty(bias_shape, dtype=self.params_dtype),
                name="bias"
            )
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.bias = None

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
            transpose_b=self.transpose_b
        )

    def construct(self, input_, weight=None):
        """
        Forward of ReplicatedLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "For ColumnParallelLinear, weight was not supplied to construct(), "
                    "and `skip_weight_param_allocation` is True."
                    )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape.
            experted_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != experted_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {experted_shape} as expected."
                )

        output = self.quant_method.apply(self, input_, weight, self.bias)
        return output

    def weight_loader(self, param, loaded_weight, loaded_shard_id: Optional[str] = None):
        """
        Load weights into the parameter, supporting both full tensor loading and sharded loading.

        Args:
            param: The target parameter to load weights into.
            loaded_weight: The weight tensor to be loaded.
            loaded_shard_id: Optional shard identifier for sharded weight loading.
                           Supported values are 'q_down' and 'kv_down' for different weight parts.
                           When None, performs full tensor loading.
        """
        offset = None
        size = None
        loaded_weight = loaded_weight[:]
        if loaded_shard_id is not None:
            if loaded_shard_id == 'q_down':
                offset = 0
                size = self.config.q_lora_rank
            if loaded_shard_id == 'kv_down':
                offset = self.config.q_lora_rank
                size = self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim
                loaded_weight = deal_linear_kv_down_weight(loaded_weight, self.config)
            if loaded_shard_id == 'gate':
                offset = 0
                size = self.config.moe_shared_expert_intermediate_size
            if loaded_shard_id == 'hidden':
                offset = self.config.moe_shared_expert_intermediate_size
                size = self.config.moe_shared_expert_intermediate_size
            if loaded_weight.shape != (size, param.shape[1]):
                raise ValueError(
                    f"'{param.name}.shape' should be equal to 'loaded_weight.shape',"
                    f" but got the shape of param is {(size, param.shape[1])} "
                    f"and the shape of weight is{loaded_weight.shape}")
            param[offset:offset + size] = ms.from_numpy(loaded_weight)
        else:
            if param.shape != loaded_weight.shape:
                raise ValueError(
                    f"'{param.name}.shape' should be equal to 'loaded_weight.shape',"
                    f" but got the shape of param is {param.shape} "
                    f"and the shape of weight is{loaded_weight.shape}")
            param.set_data(ms.from_numpy(loaded_weight))


class VocabParallelEmbedding(nn.Cell):
    """
    Embedding parallelized in the vocabulary dimension.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        config (TransformerConfig): Configuration for the transformer model.
        init_method (Union[Tensor, str, Initializer, numbers.Number]): The trainable init_method parameter. The values
            of str refer to the function `initializer`. Default: 'normal'.
        reduce_scatter_embeddings (bool): Decides whether to perform ReduceScatter after embedding lookup.
        tp_group (ProcessGroup): The process_group this linear layer used. Default: default_pgs.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            *,
            init_method: Callable,
            config: TransformerConfig,
            reduce_scatter_embeddings: bool = False,
            tp_group: ProcessGroup = default_pgs,
    ):
        super().__init__()
        if reduce_scatter_embeddings:
            raise NotImplementedError("For VocabParallelEmbedding, reduce_scatter_embeddings is not supported for now")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        quant_method = None
        # Currently does not support quantization, only use UnquantizedEmbeddingMethod.
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()
        self.quant_method: QuantizeMethodBase = quant_method
        self.tp_group = tp_group
        self.tensor_parallel_group_size = self.tp_group.size
        rank_id = self.tp_group.rank

        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, rank_id, self.tensor_parallel_group_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        self.max_index_per_partition = Tensor(self.num_embeddings_per_partition - 1, dtype=mstype.int32)
        self.expand_dims = ops.ExpandDims()
        self.gather = ops.Gather()
        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            params_dtype=config.params_dtype,
            weight_loader=self.weight_loader,
        )

    def construct(self, x):
        """
        Forward of VocabParallelEmbedding.
        Computes embeddings with optional masking and parallel reduction based on the model parallel size.
        """

        if self.tensor_parallel_group_size > 1:
            displaced_x = mint.sub(x, self.vocab_start_index)
            down_truncated_x = mint.nn.functional.relu(displaced_x)
            truncated_x = mint.minimum(down_truncated_x, self.max_index_per_partition)
            input_mask = mint.eq(displaced_x, truncated_x)
            input_mask = self.expand_dims(input_mask, -1)
        else:
            input_mask = None
            truncated_x = x
        # Get the embeddings.
        # 'embedding' has dynamic shape issue, use gather instead now.
        output_parallel = self.quant_method.embedding(self, truncated_x)
        # Mask the output embedding.
        if self.tensor_parallel_group_size > 1:
            output_parallel = mint.mul(output_parallel, input_mask)

        # Reduce across all the model parallel devices.
        output = reduce_from_model_parallel_region(output_parallel, self.tp_group)
        return output

    def _vocab_range_from_global_vocab_size(self, global_vocab_size, rank, world_size):
        if global_vocab_size % world_size != 0:
            raise ValueError(f"The vocabulary size is {global_vocab_size},"
                             f"which is not divisible by size of tensor parallel({world_size}).")
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (self.tensor_parallel_group_size, 1)
        state_dict = {}
        state_dict[self.weight.name] = {'shape': self.weight.shape,
                                        'shard': w_shard}

        return state_dict

    def weight_loader(self, param: Parameter, loaded_weight: Tensor):
        """
        Load and assign weights to corresponding parameters, supporting weight sharding loading
        in model parallel scenarios.

        Args:
            param (Parameter): Target parameter object to load weights into.
            loaded_weight (Tensor): Weight tensor loaded from checkpoint.

        """
        output_dim = getattr(param, "output_dim", None)

        if output_dim is None:
            if param.data.shape == loaded_weight.shape:
                param.set_data(loaded_weight)
                return
            raise ValueError(
                f"'{param.name}.shape' should be equal to 'loaded_weight.shape',"
                f" but got {param.data.shape} and {loaded_weight.shape}")

        # Shard indexes for loading the weight
        start_idx = self.vocab_start_index
        shard_size = self.num_embeddings_per_partition
        loaded_weight = split_loaded_weight(loaded_weight, output_dim,
                                            start_idx, shard_size)
        if loaded_weight.shape[output_dim] != shard_size:
            raise ValueError(
                f"{param.name}.shape should be equal to loaded_weight.shape,"
                f" but got the shape of weight is {loaded_weight.shape[output_dim]} and "
                f"the shape of param is {self.shard_size}"
            )

        param[:loaded_weight.shape[0]] = ms.from_numpy(loaded_weight)
        param[loaded_weight.shape[0]:] = 0


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(self, layer: nn.Cell, input_size_per_partition: int,
                       output_partition_sizes: List[int], params_dtype, **extra_weight_attrs):
        """Create weights for embedding layer."""
        weight = Parameter(mint.zeros(
            (sum(output_partition_sizes), input_size_per_partition),
            dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.insert_param_to_cell("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        self.input_size_per_partition = int(input_size_per_partition)
        self.output_size_per_partition = int(sum(output_partition_sizes))
        self.matmul = ops.MatMul(transpose_b=True)
        self.gather = ops.Gather()
        self.bias_add = ops.Add()

    def apply(self, layer: nn.Cell, x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        origin_dtype = x.dtype
        output_shape = x.shape[:-1] + (self.output_size_per_partition,)

        x = mint.reshape(x, (-1, layer.input_size))
        x = self.cast(x, layer.compute_dtype)
        weight = self.cast(weight, layer.compute_dtype)
        output_parallel = self.matmul(x, weight)

        if bias is not None and not layer.skip_bias_add:
            bias = self.cast(bias, layer.compute_dtype)
            output_parallel = mint.add(output_parallel, bias)

        output_parallel = mint.reshape(output_parallel, output_shape)
        output_parallel = self.cast(output_parallel, origin_dtype)
        return output_parallel

    def embedding(self, layer: nn.Cell, input_: Tensor) -> Tensor:
        return self.gather(layer.weight, input_, 0)
