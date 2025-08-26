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
"""Linear units for tensor parallelism."""
__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "LinearNoTP"
]

from typing import List, Optional, Callable

import mindspore._checkparam as Validator
from mindspore import nn, Tensor, ops
from mindspore.context import ParallelMode
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common import dtype
from mindspore.common.parameter import Parameter
from mindspore.ops.auto_generate import Cast, AddExt, Reshape, IndexSelect
from mindspore.ops.operations import Morph
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.init_method import init_method_zero
from mindformers.parallel_core.inference.utils import divide
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups
from mindformers.parallel_core.training_graph.device_matrix import layout


def func_infer_dtype(*args):
    """infer_dtype for Morph."""
    return args[0]


def func_infer_shape(*args):
    """infer_shape for Morph."""
    output_shape = args[0][:-1] + [args[1][0]]
    return output_shape


class VocabParallelEmbedding(nn.Cell):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.

    Args:
        num_embeddings (int): vocabulary size.
        embedding_dim (int): size of hidden state.
        init_method (str, Callable): The initialization method.
        config (TransformerConfig): The model parallel configuration.
        reduce_scatter_embeddings: Decides whether to perform ReduceScatter after embedding lookup
    """
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            init_method: Callable,
            config: TransformerConfig,
            reduce_scatter_embeddings: bool = False,
    ):
        super().__init__()
        if reduce_scatter_embeddings:
            raise NotImplementedError("For VocabParallelEmbedding, reduce_scatter_embeddings is not supported for now")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # use_cpu_initialization or perform_initialization configuration is not supported for now.
        self.weight = Parameter(init_method([self.num_embeddings, self.embedding_dim]), name="weight")
        self.gather = IndexSelect()
        self.reshape = Reshape()
        self.config = config
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        elif _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
            self.shard(config)

    def construct(self, input_):
        """Forward of vocab embedding."""
        Validator.check_type_name("input_ids", F.dtype(input_), [dtype.int32, dtype.int64], self.cls_name)
        bs, seq_len = input_.shape
        # in IndexSelect, input_ids should be 1-dimension
        input_ids_ = self.reshape(input_, (bs * seq_len,))
        # Use Gather instead of Embedding for deterministic
        # forward/backward passes via Ascend CANN operators.
        output_ = self.gather(self.weight, 0, input_ids_)
        output = self.reshape(output_, (bs, seq_len, -1))

        return output

    def shard(self, config: TransformerConfig):
        """sharding for embedding"""
        dp = 1 if config.data_parallel_size is None else config.data_parallel_size
        tp = 1 if config.tensor_model_parallel_size is None else config.tensor_model_parallel_size
        cp = 1 if config.context_parallel_size is None else config.context_parallel_size
        if config.vocab_emb_dp:
            self.gather.shard(((1, 1), (dp * cp,)))
        else:
            if self.num_embeddings % tp != 0:
                self.gather.shard(((1, 1), (dp * cp,)))
            else:
                self.gather.shard(((tp, 1), (1,)))

    def sharding_propagation(self, config: TransformerConfig):
        pass


class ColumnParallelLinear(nn.Cell):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        config (TransformerConfig): The config of the transformer model.
        bias_init (str): The initialization method for bias. Default: 'zeros'.
        bias (bool): Whether to include bias in the linear layer. Default: True.
        gather_output (bool): Whether to gather the output. Default: False.
        stride (int): The stride of the linear layer. Default: 1.
        keep_master_weight_for_test (bool): Whether to keep master weight for test. Default: False.
        skip_bias_add (bool): Whether to skip bias add. Default: False.
        skip_weight_param_allocation (bool): Whether to skip weight parameter allocation. Default: False.
        embedding_activation_buffer (List[Tensor]): The buffer for embedding activation. Default: None.
        grad_output_buffer (List[Tensor]): The buffer for gradient output. Default: None.
        is_expert (bool): Whether to use expert mode. Default: False.
        tp_comm_buffer_name (str): The name of the tensor parallel communication buffer. Default: None.
        disable_grad_reduce (bool): Whether to disable gradient reduce. Default: False.
        transpose_b (bool): Whether to transpose the weight matrix. Default: True.
        compute_dtype (dtype): The data type of the computation. Default: dtype.float16.
        init_method (Callable): The initialization method. Default: None
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: TransformerConfig,
                 init_method: Callable = None,
                 bias: bool = True,
                 gather_output: bool = False,
                 stride: int = 1,
                 keep_master_weight_for_test: bool = False,
                 skip_bias_add: bool = False,
                 skip_weight_param_allocation: bool = False,
                 embedding_activation_buffer: Optional[List[Tensor]] = None,
                 grad_output_buffer: Optional[List[Tensor]] = None,
                 is_expert: bool = False,
                 tp_comm_buffer_name: str = None,
                 disable_grad_reduce: bool = False,
                 transpose_b: bool = True,
                 bias_init: Callable = None
                 ):
        super(ColumnParallelLinear, self).__init__()
        if gather_output:
            raise NotImplementedError("For ColumnParallelLinear, `gather_output` is not supported for now")
        if stride > 1:
            raise NotImplementedError("For ColumnParallelLinear, `stride > 1` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError(
                "For ColumnParallelLinear, `keep_master_weight_for_test` is not supported for now")
        if embedding_activation_buffer is not None:
            raise NotImplementedError(
                "For ColumnParallelLinear, `embedding_activation_buffer` is not supported for now")
        if grad_output_buffer is not None:
            raise NotImplementedError("For ColumnParallelLinear, `grad_output_buffer` is not supported for now")
        if is_expert:
            raise NotImplementedError("For ColumnParallelLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For ColumnParallelLinear, `tp_comm_buffer_name` is not supported for now")
        if disable_grad_reduce:
            raise NotImplementedError("For ColumnParallelLinear, `disable_grad_reduce` is not supported for now")

        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.init_method = init_method
        self.skip_bias_add = skip_bias_add
        self.transpose_b = transpose_b
        self.skip_weight_param_allocation = skip_weight_param_allocation
        self.has_bias = bias
        self.params_dtype = config.params_dtype
        self.compute_dtype = config.compute_dtype
        self.shape = P.Shape()

        # use_cpu_initialization configuration is not supported for now.
        if skip_weight_param_allocation:
            self.weight = None
        else:
            weight_shape = (output_size, input_size) if transpose_b else (input_size, output_size)
            self.weight = Parameter(init_method(weight_shape), name='weight')

        if self.has_bias:
            if bias_init is None:
                bias_init = init_method_zero(self.params_dtype)
            self.bias = Parameter(bias_init((output_size,)), name='bias')
        else:
            self.bias = None

        self.cast = Cast()
        self.matmul = ops.MatMul(transpose_b=transpose_b)
        if not skip_bias_add:
            self.add = AddExt()
        self.reshape = Reshape()

        # init morphed layer
        self.morphed_forward_with_bias = Morph(self.forward_func_with_bias,
                                               func_infer_shape,
                                               func_infer_dtype).add_prim_attr("self_define_shard", True)
        self.morphed_forward = Morph(self.forward_func, func_infer_shape, func_infer_dtype).add_prim_attr(
            "self_define_shard", True)

        if config is not None:
            if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
                self.sharding_propagation(config)

            if _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
                self.shard()

    def construct(self, input_: Tensor, weight: Tensor = None) -> tuple[Tensor, Tensor]:
        """Forward of ColumnParallelLinear.

        Args:
            input_ (Tensor): The input tensor.
            weight (Tensor): The weight tensor. Default: None.

        Returns:
            - output (Tensor): The output tensor.
            - bias (Tensor): The bias
        """
        if weight is None:
            if self.skip_weight_param_allocation:
                raise ValueError("For ColumnParallelLinear, when `skip_weight_param_allocation` is enabled,"
                                 " `weight` is required, but got None")
            weight = self.weight
        weight = self.cast(weight, self.compute_dtype)

        if not self.skip_bias_add and self.has_bias:
            bias = self.cast(self.bias, self.compute_dtype)
            output = self.morphed_forward_with_bias(input_, weight, bias)
            bias = None
        else:
            output = self.morphed_forward(input_, weight)
            bias = self.bias

        return output, bias

    def forward_func_with_bias(self, input_, weight, bias):
        """Morphed forward."""
        output_size = int(self.shape(weight)[0] if self.transpose_b else self.shape(weight)[-1])
        output_shape = input_.shape[:-1] + (output_size,)
        input_ = self.reshape(input_, (-1, self.input_size))

        ori_dtype = input_.dtype

        weight = self.cast(weight, self.compute_dtype)
        input_ = self.cast(input_, self.compute_dtype)

        input_ = self.matmul(input_, weight)

        bias = self.cast(bias, self.compute_dtype)
        input_ = self.add(input_, bias)

        input_ = self.cast(input_, ori_dtype)
        output = self.reshape(input_, output_shape)
        return output

    def forward_func(self, input_, weight):
        """Morphed forward."""
        output_size = int(self.shape(weight)[0] if self.transpose_b else self.shape(weight)[-1])
        output_shape = input_.shape[:-1] + (output_size,)
        input_ = self.reshape(input_, (-1, self.input_size))

        ori_dtype = input_.dtype

        weight = self.cast(weight, self.compute_dtype)
        input_ = self.cast(input_, self.compute_dtype)

        input_ = self.matmul(input_, weight)

        input_ = self.cast(input_, ori_dtype)
        output = self.reshape(input_, output_shape)
        return output

    def shard(self) -> None:
        """Shard the operators in ColumnParallelLinear.
        """
        self.morphed_forward_with_bias.shard(
            in_strategy=(
                layout("cp", "dp", "None"),        # input_       [S, B, h]
                layout("tp", "None"),              # weight       [H, h]
                layout("tp"),                      # bias         [H]
            ),
            out_strategy=(
                layout("cp", "dp", "tp"),          # output       [S, B, H]
            )
        )

        self.morphed_forward.shard(
            in_strategy=(
                layout("cp", "dp", "None"),        # input_       [S, B, h]
                layout("tp", "None"),              # weight       [H, h]
            ),
            out_strategy=(
                layout("cp", "dp", "tp"),          # output       [S, B, H]
            )
        )


    def sharding_propagation(self, config: TransformerConfig) -> None:
        """Shard the operators in ColumnParallelLinear in sharding propagation mode.

        Args:
            config (TransformerConfig): The config of the transformer model.
        """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1

        if self.transpose_b:
            weight_strategy = (tp, 1)
        else:
            weight_strategy = (1, tp)
        matmul_in_strategy = ((dp * cp, 1), weight_strategy)
        self.matmul.shard(in_strategy=matmul_in_strategy)


class RowParallelLinear(nn.Cell):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X along
    its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        config (TransformerConfig): The config of the transformer model.
        bias_init (str): The initialization method for bias. Default: 'zeros'.
        bias (bool): Whether to include bias in the linear layer. Default: True.
        input_is_parallel (bool): Whether the input is already parallelized. Default: False.
        skip_bias_add (bool): Whether to skip bias add. Default: False.
        stride (int): The stride of the linear layer. Default: 1.
        keep_master_weight_for_test (bool): Whether to keep master weight for test. Default: False.
        is_expert (bool): Whether to use expert mode. Default: False.
        tp_comm_buffer_name (str): The name of the tensor parallel communication buffer. Default: None.
        transpose_b (bool): Whether to transpose the weight matrix. Default: True.
        compute_dtype (dtype): The data type of the computation. Default: dtype.float16.
        init_method (Callable): The initialization method. Default: None
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: TransformerConfig,
                 init_method: Callable = None,
                 bias: bool = True,
                 input_is_parallel: bool = False,
                 skip_bias_add: bool = False,
                 stride: int = 1,
                 keep_master_weight_for_test: bool = False,
                 is_expert: bool = False,
                 tp_comm_buffer_name: str = None,
                 transpose_b: bool = True,
                 bias_init: Callable = None
                 ):
        super(RowParallelLinear, self).__init__()
        if input_is_parallel:
            raise NotImplementedError("For RowParallelLinear, `input_is_parallel` is not supported for now")
        if stride > 1:
            raise NotImplementedError("For RowParallelLinear, `stride > 1` is not supported for now")
        if keep_master_weight_for_test:
            raise NotImplementedError("For RowParallelLinear, `keep_master_weight_for_test` is not supported for now")
        if is_expert:
            raise NotImplementedError("For RowParallelLinear, `is_expert` is not supported for now")
        if tp_comm_buffer_name is not None:
            raise NotImplementedError("For RowParallelLinear, `tp_comm_buffer_name` is not supported for now")

        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.init_method = init_method
        self.transpose_b = transpose_b
        self.skip_bias_add = skip_bias_add
        self.compute_dtype = config.compute_dtype
        self.params_dtype = config.params_dtype
        self.has_bias = bias
        self.sequence_parallel = config.sequence_parallel
        self.tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1
        if self.tp != 1:
            self.model_comm_pgs = ModelCommProcessGroups.use_parallel_state_groups(required_groups=['tp'])
            self.tp_group = self.model_comm_pgs.tp
            self.reduce_scatter = ops.ReduceScatter(group=self.tp_group.group)
            self.all_reduce = ops.AllReduce(group=self.tp_group.group)
        self.input_size_per_partition = divide(input_size, self.tp)
        self.shape = P.Shape()

        # use_cpu_initialization configuration is not supported for now.
        weight_shape = (output_size, input_size) if transpose_b else (input_size, output_size)
        self.weight = Parameter(init_method(weight_shape), name='weight')

        if self.has_bias:
            if bias_init is None:
                bias_init = init_method_zero(self.params_dtype)
            self.bias = Parameter(bias_init((output_size,)), name='bias')
        else:
            self.bias = None

        self.cast = Cast()
        self.matmul = ops.MatMul(transpose_b=transpose_b)

        if not skip_bias_add:
            self.add = AddExt()
        self.reshape = Reshape()

        # init morphed layer
        self.morphed_forward_with_bias = Morph(self.forward_func_with_bias,
                                               func_infer_shape,
                                               func_infer_dtype).add_prim_attr("self_define_shard", True)
        self.morphed_forward = Morph(self.forward_func, func_infer_shape, func_infer_dtype).add_prim_attr(
            "self_define_shard", True)

        if config is not None:
            if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
                self.sharding_propagation(config)

            if _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL,):
                self.shard()

    def construct(self, input_: Tensor) -> tuple[Tensor, Tensor]:
        """Forward of RowParallelLinear.

        Args:
            input_ (Tensor): The input tensor.

        Returns:
            - output (Tensor): The output tensor.
            - bias (Tensor): The bias
        """
        weight = self.cast(self.weight, self.compute_dtype)
        if not self.skip_bias_add and self.has_bias:
            bias = self.cast(self.bias, self.compute_dtype)
            output = self.morphed_forward_with_bias(input_, weight, bias)
            bias = None
        else:
            output = self.morphed_forward(input_, weight)
            bias = self.bias
        return output, bias

    def forward_func_with_bias(self, input_, weight, bias):
        """Morphed forward."""
        output_size = int(self.shape(weight)[0] if self.transpose_b else self.shape(weight)[-1])
        output_shape = input_.shape[:-1] + (output_size,)
        if self.sequence_parallel:
            output_shape = (output_shape[0] // self.tp,) + output_shape[1:]
        input_ = self.reshape(input_, (-1, self.input_size_per_partition))

        ori_dtype = input_.dtype
        weight = self.cast(weight, self.compute_dtype)
        input_ = self.cast(input_, self.compute_dtype)

        input_ = self.matmul(input_, weight)
        if self.tp != 1:
            if self.sequence_parallel:
                input_ = self.reduce_scatter(input_)
            else:
                input_ = self.all_reduce(input_)

        if not self.skip_bias_add and self.has_bias:
            bias = self.cast(bias, self.compute_dtype)
            input_ = self.add(input_, bias)

        input_ = self.cast(input_, ori_dtype)
        output = self.reshape(input_, output_shape)
        return output

    def forward_func(self, input_, weight):
        """Morphed forward."""
        output_size = int(self.shape(weight)[0] if self.transpose_b else self.shape(weight)[-1])
        output_shape = input_.shape[:-1] + (output_size,)
        if self.sequence_parallel:
            output_shape = (output_shape[0] // self.tp,) + output_shape[1:]
        input_ = self.reshape(input_, (-1, self.input_size_per_partition))

        ori_dtype = input_.dtype
        weight = self.cast(weight, self.compute_dtype)
        input_ = self.cast(input_, self.compute_dtype)

        input_ = self.matmul(input_, weight)
        if self.tp != 1:
            if self.sequence_parallel:
                input_ = self.reduce_scatter(input_)
            else:
                input_ = self.all_reduce(input_)

        input_ = self.cast(input_, ori_dtype)
        output = self.reshape(input_, output_shape)
        return output

    def shard(self) -> None:
        """Shard the operators in RowParallelLinear.
        """
        if self.sequence_parallel:
            self.morphed_forward_with_bias.shard(
                in_strategy=(
                    layout("cp", "dp", "tp"),            # input_,      [S, B, h]
                    layout("None", "tp"),                # weight       [H, h]
                    layout("None"),                      # bias         [H]
                ),
                out_strategy=(
                    layout("cp_tp", "dp", "None"),  # output       [S, B, H]
                )
            )
            self.morphed_forward.shard(
                in_strategy=(
                    layout("cp", "dp", "tp"),            # input_,      [S, B, h]
                    layout("None", "tp"),                # weight       [H, h]
                ),
                out_strategy=(
                    layout("cp_tp", "dp", "None"),  # output       [S, B, H]
                )
            )
        else:
            self.morphed_forward_with_bias.shard(
                in_strategy=(
                    layout("cp", "dp", "tp"),            # input_,      [S, B, h]
                    layout("None", "tp"),                # weight       [H, h]
                    layout("None"),                      # bias         [H]
                ),
                out_strategy=(
                    layout("cp", "dp", "None"),          # output       [S, B, H]
                )
            )
            self.morphed_forward.shard(
                in_strategy=(
                    layout("cp", "dp", "tp"),            # input_,      [S, B, h]
                    layout("None", "tp"),                # weight       [H, h]
                ),
                out_strategy=(
                    layout("cp", "dp", "None"),          # output       [S, B, H]
                )
            )


    def sharding_propagation(self, config: TransformerConfig) -> None:
        """Shard the operators in RowParallelLinear in sharding propagation mode.

        Args:
            config (TransformerConfig): The config of the transformer model.
        """
        dp = config.data_parallel_size if config.data_parallel_size is not None else 1
        cp = config.context_parallel_size if config.context_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config.tensor_model_parallel_size is not None else 1

        if self.transpose_b:
            weight_strategy = (1, tp)
        else:
            weight_strategy = (tp, 1)
        matmul_in_strategy = ((dp * cp, tp), weight_strategy)
        self.matmul.shard(in_strategy=matmul_in_strategy)


class LinearNoTP(ColumnParallelLinear):
    """Linear layer without tensor parallelism.

    The linear layer is defined as Y = XA + b. A is not parallelized.

    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        config (TransformerConfig): The config of the transformer model.
        bias_init (str): The initialization method for bias. Default: 'zeros'.
        bias (bool): Whether to include bias in the linear layer. Default: True.
        gather_output (bool): Whether to gather the output. Default: False.
        stride (int): The stride of the linear layer. Default: 1.
        keep_master_weight_for_test (bool): Whether to keep master weight for test. Default: False.
        skip_bias_add (bool): Whether to skip bias add. Default: False.
        skip_weight_param_allocation (bool): Whether to skip weight parameter allocation. Default: False.
        embedding_activation_buffer (List[Tensor]): The buffer for embedding activation. Default: None.
        grad_output_buffer (List[Tensor]): The buffer for gradient output. Default: None.
        is_expert (bool): Whether to use expert mode. Default: False.
        tp_comm_buffer_name (str): The name of the tensor parallel communication buffer. Default: None.
        disable_grad_reduce (bool): Whether to disable gradient reduce. Default: False.
        transpose_b (bool): Whether to transpose the weight matrix. Default: True.
        init_method (Callable): The initialization method. Default: None
    """

    def shard(self) -> None:
        """Shard the operators in LinearNoTP."""
        self.morphed_forward_with_bias.shard(
            in_strategy=(
                layout("cp", "dp", "None"),  # input_       [S, B, h]
                layout("None", "None"),  # weight       [H, h]
                layout("None"),  # bias         [H]
            ),
            out_strategy=(
                layout("cp", "dp", "None"),  # output       [S, B, H]
            )
        )

        self.morphed_forward.shard(
            in_strategy=(
                layout("cp", "dp", "None"),  # input_       [S, B, h]
                layout("None", "None"),  # weight       [H, h]
            ),
            out_strategy=(
                layout("cp", "dp", "None"),  # output       [S, B, H]
            )
        )
