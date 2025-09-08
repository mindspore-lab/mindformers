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
"""Gemm linear units for tensor parallelism"""

__all__ = [
    'UnquantizedGroupedLinearMethod',
    'ColumnParallelGroupedLinear',
    'RowParallelGroupedLinear',
]

from typing import Optional
from abc import abstractmethod

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import Tensor, Parameter, nn, ops, mint

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.quantization import QuantizationConfig
from mindformers.parallel_core.inference.quantization.base_config import QuantizeMethodBase
from mindformers.parallel_core.inference.utils import divide
from mindformers.parallel_core.inference.weights_utils import set_weight_attrs
from mindformers.parallel_core.inference.tensor_parallel.mappings import (
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region
)
from mindformers.parallel_core.inference.parallel_state import ProcessGroup, default_pgs
from mindformers.parallel_core.inference.weights_utils import split_loaded_weight, cpu_offload_weights_params


class GroupedLinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) grouped linear methods."""

    @abstractmethod
    def create_weights(self, layer: nn.Cell, num_local_experts: int,
                       input_size_per_partition: int, output_size_per_partition: int,
                       params_dtype, **extra_weight_attrs):
        """Create weights for a grouped linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the GroupedLinearMethodBase factory.
            num_local_experts: The number of local experts.
            input_size_per_partition: Sizes of the input dim on rank X.
            output_size_per_partition: Sizes of the output dim on rank X.
            params_dtype: Datatype of the parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self,
              layer: nn.Cell,
              x: Tensor,
              weight: Tensor,
              bias: Optional[Tensor] = None,
              group_list: Tensor = None) -> Tensor:
        """
        Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer.
        """

        raise NotImplementedError


class UnquantizedGroupedLinearMethod(GroupedLinearMethodBase):
    """Grouped linear method without quantization."""

    def __init__(self):
        self.cast = P.Cast()
        self.matmul = ops.auto_generate.GroupedMatmulV4()

    def create_weights(self, layer: nn.Cell, num_local_experts: int,
                       input_size_per_partition: int, output_partition_sizes: list[int],
                       params_dtype, **extra_weight_attrs):

        weight = Parameter(
            mint.zeros(
                (num_local_experts,
                 input_size_per_partition,
                 sum(output_partition_sizes)),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 2})
        if layer is not None:
            layer.insert_param_to_cell("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        return weight

    def apply(self,
              layer: nn.Cell,
              x: Tensor,
              weight: Tensor,
              bias: Parameter = None,
              group_list: Tensor = None) -> Tensor:
        origin_dtype = x.dtype
        x = self.cast(x, layer.compute_dtype)
        weight = self.cast(weight, layer.compute_dtype)
        output_parallel = self.matmul([x], [weight], None, None, None, None, None, None,
                                      group_list, split_item=3, group_type=0, group_list_type=1)[0]

        if bias is not None:
            bias = self.cast(bias, layer.compute_dtype)
            output_parallel = mint.add(output_parallel, bias)

        output_parallel = self.cast(output_parallel, origin_dtype)
        return output_parallel


class GroupedLinearBase(nn.Cell):
    """Base grouped linear layer.

    Args:
        num_local_experts (int): The number of local expert.
        input_size (int): input dimension of the linear layer.
        output_size (int): output dimension of the linear layer.
        skip_bias_add (bool): If true, skip adding bias but instead return it. Default: False.
        params_dtype (mstype): Data type for the parameters. Default: mstype.float32
        quant_config (QuantizationConfig, optional): Quantization configuration. Default: None.
        prefix (str): The prefix string for this linear layer. Default: empty string("").
    """

    def __init__(
            self,
            num_local_experts: int,
            input_size: int,
            output_size: int,
            skip_bias_add: bool = False,
            params_dtype: mstype = mstype.float32,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = ""
    ):
        super().__init__()

        # Keep input parameters
        self.num_local_experts = num_local_experts
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.quant_method: Optional[
            QuantizeMethodBase] = UnquantizedGroupedLinearMethod()
        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

    def construct(self, x: Tensor, weight: Tensor, group_list: Tensor) -> Tensor:
        raise NotImplementedError


class ColumnParallelGroupedLinear(GroupedLinearBase):
    r"""
    The group linear layer with weight sliced on second dimension by tensor parallel size.
    This layer implements the operation as:

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

    where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
    and :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True).

    Args:
        num_local_experts (int): The number of local expert.
        input_size (int): The number of channels in the input space.
        output_size (int): The number of channels in the output space.
        config (TransformerConfig): Transformer configuration.
        bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        gather_output (bool): Specifies whether gather the output on each tensor parallel rank. Default: False.
        stride (int): Stride parameter, currently unsupported for `stride > 1`. Default: 1.
        skip_bias_add (bool): This was added to enable performance optimizations where bias can be fused with other
            element-wise operations. We skip adding bias but instead return it. Default: False.
        weight (Tensor): Use externally passed weights to skip the process of creating initial weights. Default: None.
        is_expert (bool): Specifies whether this linear layer is an expert. Default: True.
        tp_comm_buffer_name (str): Tensor-parallel communication buffer name, currently unsupported. Default: None.
        tp_group (ProcessGroup): The process_group this linear layer used. Default: default_pgs.
        quant_config (QuantizationConfig, optional): Quantization configuration. Default: None.
        prefix (str): The prefix string for this linear layer. Default: empty string("").

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
            num_local_experts: int,
            input_size: int,
            output_size: int,
            *,
            config: TransformerConfig,
            bias: bool = False,
            gather_output: bool = False,
            stride: int = 1,
            skip_bias_add: bool = False,
            weight: Tensor = None,
            is_expert: bool = True,
            tp_comm_buffer_name: str = None,
            tp_group: ProcessGroup = default_pgs,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = ""
    ):
        super(ColumnParallelGroupedLinear, self).__init__(num_local_experts,
                                                          input_size,
                                                          output_size,
                                                          skip_bias_add,
                                                          config.params_dtype,
                                                          quant_config=quant_config,
                                                          prefix=prefix)
        if stride > 1:
            raise NotImplementedError(
                "For ColumnParallelGroupedLinear, `stride > 1` is not supported for now, "
                "but got `stride={}`".format(stride))
        if skip_bias_add:
            raise NotImplementedError(
                "For ColumnParallelGroupedLinear, `skip_bias_add=True` is not supported for now."
            )
        if not is_expert:
            raise NotImplementedError(
                "For ColumnParallelGroupedLinear, `is_expert=False` is not supported for now.")
        if tp_comm_buffer_name:
            raise NotImplementedError(
                "For ColumnParallelGroupedLinear, `tp_comm_buffer_name` is not supported for now."
            )

        self.config = config
        self.has_bias = bias
        self.gather_output = gather_output
        self.skip_weight_param_allocation = weight is not None
        self.compute_dtype = config.compute_dtype

        # tp_group passed in here is model_comm_pgs.moe_tp
        self.tp_group = tp_group
        self.tensor_parallel_group_size = self.tp_group.size
        self.output_size_per_partition = divide(output_size, self.tensor_parallel_group_size)
        if self.quant_method is None:
            raise ValueError("`quant_method` is not initialized in ColumnParallelGroupedLinear.")

        # create weights other than experts weights, like quantization weight scale and offset.
        self.quant_method.create_weights(
            layer=self,
            num_local_experts=self.num_local_experts,
            input_size_per_partition=self.input_size,
            output_partition_sizes=[self.output_size_per_partition],
            params_dtype=self.config.params_dtype,
            skip_weight_param_allocation=self.skip_weight_param_allocation,
            weight_loader=self.weight_loader
        )

        if self.skip_weight_param_allocation:
            self.weight = weight

        if self.has_bias:
            bias_shape = (self.num_local_experts, self.output_size_per_partition)
            self.bias = Parameter(
                mint.empty(bias_shape, dtype=self.params_dtype),
                name="bias"
            )
        else:
            self.bias = None

    def construct(self, input_parallel, weight=None, group_list=None):
        """Forward of ColumnParallelGroupedLinear."""
        if weight is None:
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape.
            expected_shape = (self.num_local_experts, self.input_size, self.output_size_per_partition)
            if weight.shape != expected_shape:
                raise ValueError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected."
                )
        output_parallel = self.quant_method.apply(self, input_parallel, weight, self.bias, group_list)
        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel, self.tp_group)
        else:
            output = output_parallel
        return output

    def sharded_state_dict(self):
        """Provide the sharded state dict."""
        expert_parallel_group_size = self.config.num_moe_experts // self.num_local_experts
        w_shard = (expert_parallel_group_size, 1, self.tensor_parallel_group_size)

        state_dict = {}
        if not self.skip_weight_param_allocation:
            state_dict[self.weight.name] = {'shape': self.weight.shape,
                                            'shard': w_shard}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (expert_parallel_group_size, self.tensor_parallel_group_size)}
        return state_dict

    def weight_loader(self, param, loaded_weight, shard_id, expert_id) -> None:
        """
        Args:
            param: The parameter tensor in the model, used to store the loaded weights.
            loaded_weight: The weight data loaded from a file or checkpoint.
            shard_id: The weight shard identifier, such as "w1" or "w3",
                        indicating which part of the weights is currently being processed.
            expert_id: The index of the expert network, used to locate the parameters of a specific expert
                        in the MoE structure.

        Returns:
            None. This function directly modifies the input `param` parameter.
        """
        weight_is_3d = expert_id is None
        weight_needs_transpose = self._determine_transpose_need(loaded_weight, weight_is_3d, param)

        if shard_id is None:
            self._load_full_weight(param, loaded_weight, weight_is_3d)
            return

        self._load_sharded_weight(param,
                                  loaded_weight,
                                  shard_id,
                                  expert_id,
                                  weight_is_3d,
                                  weight_needs_transpose)
        cpu_offload_weights_params(param, self.config.cpu_offloading_weights)

    def _determine_transpose_need(self, loaded_weight, weight_is_3d, param):
        """Determine if the loaded weight needs transposition."""
        if (weight_is_3d and len(loaded_weight.get_shape()) == 2) or \
                (not weight_is_3d and len(loaded_weight.get_shape()) == 1):
            return False
        return True

    def _load_full_weight(self, param, loaded_weight, weight_is_3d):
        """Load full weight without sharding."""
        if not weight_is_3d or loaded_weight.get_shape() != param.shape:
            raise ValueError(
                f"Expected loaded weight to be 3-dimensional with shape {param.shape}, "
                f"but got {loaded_weight.get_shape()}."
            )
        param.set_data(ms.from_numpy(loaded_weight))

    def _load_sharded_weight(self, param, loaded_weight, shard_id, expert_id, weight_is_3d, weight_needs_transpose):
        """Load sharded weight for w1 or w3."""
        # Handle modelslim weight_scale adaptation
        if not weight_is_3d and param.name.endswith("w_scale") and len(loaded_weight.get_shape()) == 2 \
                and loaded_weight.get_shape()[1] == 1:
            loaded_weight = loaded_weight[:].squeeze(-1)
            weight_needs_transpose = False

        param_output_dim = getattr(param, "output_dim", None)
        shard_size = param.shape[param_output_dim] // 2
        shard_dim = self._get_shard_dim(param, weight_needs_transpose, param_output_dim, weight_is_3d)

        tp_rank = self.tp_group.rank
        start_idx = tp_rank * shard_size
        loaded_weight = split_loaded_weight(loaded_weight, shard_dim, start_idx, shard_size)

        if weight_needs_transpose:
            loaded_weight = loaded_weight.T

        expected_shape = self._get_expected_shape(param, shard_size, weight_is_3d, param_output_dim)
        if loaded_weight.shape != expected_shape:
            raise ValueError(
                f"'param.data.shape' should be equal to 'loaded_weight.get_shape()',"
                f" but got the shape of param is {expected_shape} and "
                f"the shape of weight is{loaded_weight.shape}")

        update_indices = self._get_update_indices(param, expert_id, shard_id, shard_size, param_output_dim,
                                                  weight_is_3d)
        param.init_data()
        if param.dtype == ms.qint4x2 or param.dtype == ms.uint64:
            param.asnumpy()[tuple(update_indices)] = loaded_weight
        else:
            param[tuple(update_indices)] = ms.from_numpy(loaded_weight)

    def _get_shard_dim(self, param, weight_needs_transpose, param_output_dim, weight_is_3d):
        """Determine the shard dimension for splitting the loaded weight."""
        shard_dim = getattr(param, "input_dim", None) if weight_needs_transpose else param_output_dim
        if not weight_is_3d:
            shard_dim -= 1  # Remove the expert dimension for 2D weights.
        return shard_dim

    def _get_expected_shape(self, param, shard_size, weight_is_3d, param_output_dim):
        """Get the expected shape for the loaded weight shard."""
        expected_shape = list(param.shape)
        expected_shape[param_output_dim] = shard_size
        if not weight_is_3d:
            expected_shape = expected_shape[1:]  # Remove the expert dimension for 2D weights.
        return tuple(expected_shape)

    def _get_update_indices(self, param, expert_id, shard_id, shard_size, param_output_dim, weight_is_3d):
        """Get the indices for updating the parameter with the loaded weight shard."""
        update_indices = [slice(None)] * len(param.shape)
        if not weight_is_3d:
            update_indices[0] = expert_id  # Update only the specific expert's weight.
        if shard_id == "w1":
            update_indices[param_output_dim] = slice(None, shard_size)
        elif shard_id == "w3":
            update_indices[param_output_dim] = slice(shard_size, 2 * shard_size)
        return update_indices


class RowParallelGroupedLinear(GroupedLinearBase):
    r"""
    The group linear layer with weight sliced on first dimension by tensor parallel size.
    This layer implements the operation as:

    .. math::
        \text{outputs} = \text{inputs} * \text{weight} + \text{bias},

    where :math:`inputs` is the input tensors, :math:`\text{weight}` is a weight matrix created by the layer,
    and :math:`\text{bias}` is a bias vector created by the layer (only if has_bias is True).

    Args:
        num_local_experts (int): The number of local expert.
        input_size (int): The number of channels in the input space.
        output_size (int): The number of channels in the output space.
        config (TransformerConfig): Transformer configuration.
        bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        input_is_parallel (bool): Specifies whether the input tensor has already been sliced on last dimension.
            Default: True
        skip_bias_add (bool): This was added to enable performance optimizations where bias can be fused with other
            element-wise operations. We skip adding bias but instead return it. Default: False.
        weight (Tensor): Use externally passed weights to skip the process of creating initial weights. Default: None.
        stride (int): Stride parameter, currently unsupported for `stride > 1`. Default: 1.
        delay_allreduce (bool): Whether to delay allreduce during forward function. Default: True
        is_expert (bool): Specifies whether this linear layer is an expert. Default: True.
        tp_comm_buffer_name (str): Tensor-parallel communication buffer name, currently unsupported. Default: None.
        tp_group (ProcessGroup): The process_group this linear layer used. Default: default_pgs.
        quant_config (QuantizationConfig, optional): Quantization configuration. Default: None.
        prefix (str): The prefix string for this linear layer. Default: empty string("").

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
            num_local_experts: int,
            input_size: int,
            output_size: int,
            *,
            config: TransformerConfig,
            bias: bool = False,
            input_is_parallel: bool = True,
            skip_bias_add: bool = False,
            weight: Tensor = None,
            stride: int = 1,
            delay_allreduce: bool = True,
            is_expert: bool = True,
            tp_comm_buffer_name: str = None,
            tp_group: ProcessGroup = default_pgs,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = ""
    ):
        super(RowParallelGroupedLinear, self).__init__(num_local_experts,
                                                       input_size,
                                                       output_size,
                                                       skip_bias_add,
                                                       config.params_dtype,
                                                       quant_config=quant_config,
                                                       prefix=prefix)
        if stride > 1:
            raise NotImplementedError(
                "For RowParallelGroupedLinear, `stride > 1` is not supported for now, "
                "but got `stride={}`".format(stride))
        if not is_expert:
            raise NotImplementedError(
                "For RowParallelGroupedLinear, `is_expert=False` is not supported for now.")
        if tp_comm_buffer_name:
            raise NotImplementedError(
                "For RowParallelGroupedLinear, `tp_comm_buffer_name` is not supported for now.")
        if delay_allreduce and bias:
            raise RuntimeError(
                "In RowParallelGroupedLinear, `delay_allreduce` and `bias` cannot be enabled simultaneously, "
                "otherwise the accuracy will be incorrect")

        self.config = config
        self.has_bias = bias
        self.input_is_parallel = input_is_parallel
        self.delay_allreduce = delay_allreduce
        self.skip_weight_param_allocation = weight is not None
        self.compute_dtype = config.compute_dtype

        # tp_group passed in here is model_comm_pgs.moe_tp
        self.tp_group = tp_group
        self.tensor_parallel_group_size = self.tp_group.size
        self.input_size_per_partition = divide(input_size, self.tensor_parallel_group_size)
        if self.quant_method is None:
            raise ValueError("`quant_method` is not initialized in RowParallelGroupedLinear.")

        self.quant_method.create_weights(
            layer=self,
            num_local_experts=self.num_local_experts,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[self.output_size],
            params_dtype=self.config.params_dtype,
            skip_weight_param_allocation=self.skip_weight_param_allocation,
            weight_loader=self.weight_loader
        )
        if self.skip_weight_param_allocation:
            self.weight = weight

        if self.has_bias:
            bias_shape = (self.num_local_experts, self.output_size)
            self.bias = Parameter(
                mint.empty(bias_shape, dtype=self.params_dtype),
                name="bias"
            )
        else:
            self.bias = None

    def construct(self, input_, weight=None, group_list=None):
        """Forward of RowParallelGroupedLinear."""
        if weight is None:
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape.
            expected_shape = (self.num_local_experts, self.input_size_per_partition, self.output_size)
            if weight.shape != expected_shape:
                raise ValueError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected."
                )

        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_, self.tp_group)

        bias_ = None if self.tp_group.rank > 0 else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, weight, bias_, group_list)
        if self.delay_allreduce or self.skip_bias_add:
            output = output_parallel
        else:
            output = reduce_from_model_parallel_region(output_parallel, self.tp_group)
        return output

    def sharded_state_dict(self):
        """Provide the sharded state dict."""
        expert_parallel_group_size = self.config.num_moe_experts // self.num_local_experts
        w_shard = (expert_parallel_group_size, self.tensor_parallel_group_size, 1)

        state_dict = {}
        if not self.skip_weight_param_allocation:
            state_dict[self.weight.name] = {'shape': self.weight.shape,
                                            'shard': w_shard}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (expert_parallel_group_size, 1)}
        return state_dict

    def weight_loader(self, param, loaded_weight, shard_id, expert_id) -> None:
        """
        Load and process weights for RowParallelGroupedLinear layer.

        This method handles the loading of weights that have been partitioned along the input dimension
        according to tensor parallelism for grouped linear layers in MoE (Mixture of Experts) architecture.
        Each expert's weights are loaded and processed separately based on the shard ID and expert ID.

        Args:
            param: The parameter tensor in the model to store the loaded weights.
                  For RowParallelGroupedLinear, this is typically a 3D tensor with shape
                  [num_experts, input_size_per_partition, output_size].
            loaded_weight: The weight data loaded from checkpoint file.
                          Shape depends on whether it's a full load or partitioned load.
            shard_id: The weight shard identifier, specifically "w2" for RowParallelGroupedLinear,
                      indicating which part of the weights is being processed.
            expert_id: The index of the expert network, used to locate the parameters of a specific expert
                       in the MoE structure.
        """
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        shard_dim = getattr(param, "input_dim", None)

        if not param.name.endswith("weight") and shard_dim is None:
            # adapter for modelslim weight, weight_scale is (oc, 1)  not (oc)
            if param.name.endswith("w_scale") and len(loaded_weight.get_shape()) == 2 \
               and loaded_weight.get_shape()[1] == 1:
                loaded_weight = loaded_weight[:].squeeze(-1)
            param.init_data()
            param[expert_id] = ms.from_numpy(loaded_weight[:])
            return
        shard_id_to_sharded_dim = {"w2": 0}
        shard_id_to_sharded_dim_int4 = {"w2": -1}
        shard_dim = shard_id_to_sharded_dim.get(shard_id)
        shard_dim_int4 = shard_id_to_sharded_dim_int4.get(shard_id)
        full_load = len(loaded_weight.get_shape()) == 3
        if full_load:
            shard_dim += 1

        param.init_data()

        # Case model weights
        if param.dtype == ms.qint4x2:
            shard_size = loaded_weight.get_shape()[shard_dim_int4] // self.tensor_parallel_group_size
        else:
            expert_data = param.data if full_load else param.data[expert_id]
            shard_size = expert_data.shape[shard_dim]

        # Because this weight shape is two-dimensional,
        # but the dimension splitting in the network is defined based on three dimensions,
        # so the splitting dimension needs to be subtracted by 1.
        output_dim = getattr(param, "output_dim", None) - 1
        tp_rank = self.tp_group.rank
        start_idx = tp_rank * shard_size
        # The Hugging Face weight shape is [hidden_size, moe_ffn_hidden_size]
        # The shape of param is [moe_ffn_hidden_size, hidden_size]
        # So must be transposed.
        loaded_weight = split_loaded_weight(loaded_weight, output_dim, start_idx, shard_size).T
        if not loaded_weight.shape:
            loaded_weight = loaded_weight.reshape(1)
        if loaded_weight.shape == (shard_size, param.shape[2]):
            param.init_data()
            if param.dtype == ms.qint4x2:
                param.asnumpy()[expert_id] = loaded_weight
            else:
                param[expert_id] = ms.from_numpy(loaded_weight)
        else:
            raise ValueError(
                f"'param.data.shape' should be equal to 'loaded_weight.shape',"
                f" but got the shape of param is {(shard_size, param.shape[2])} and "
                f"the shape of weight is{loaded_weight.shape}")
        cpu_offload_weights_params(param, self.config.cpu_offloading_weights)
