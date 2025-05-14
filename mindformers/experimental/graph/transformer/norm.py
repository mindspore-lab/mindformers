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
"""Normalization"""
import mindspore as ms
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype

from mindspore import nn, Parameter
from mindspore.context import ParallelMode
from mindspore.ops.auto_generate import MeanExt, Sqrt, Rsqrt, SubExt, AddExt, Mul, Div, Cast
from mindspore.common.initializer import initializer
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig

__all__ = ["get_norm", "Norm", "FusedNorm"]


def get_strategy(config):
    """Retrieves the parallel strategy"""
    dp = 1 if config.data_parallel is None else config.data_parallel
    tp = 1 if config.tensor_parallel is None else config.tensor_parallel
    cp = 1 if config.context_parallel is None else config.context_parallel
    return (dp, tp, cp)


class LayerNorm(nn.Cell):
    r"""
    Layer norm operation.

    Args:
        normalized_shape (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, config, dim, eps=1e-5, param_init_type=mstype.float32, layernorm_compute_type=mstype.float32):
        super(LayerNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'param_init_type' should be in [float32, float16, bfloat16], "
                            "but got the type : {}.".format(type(param_init_type)))
        if layernorm_compute_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'layernorm_compute_type' should be in [float32, float16, bfloat16], "
                            "but got the type : {}.".format(type(layernorm_compute_type)))

        self.gamma = Parameter(initializer('ones', dim, param_init_type), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', dim, param_init_type), name="beta",
                              parallel_optimizer=False)

        self.mean = MeanExt()
        self.mean2 = MeanExt()
        self.square = P.Square()
        self.sqrt = Sqrt()
        self.sub = SubExt()
        self.add = AddExt()
        self.eps = eps
        self.mul = Mul()
        self.add2 = AddExt()
        self.real_div = Div()
        self.compute_type = layernorm_compute_type
        self.cast = P.Cast()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        else:
            self.shard(config)

    def construct(self, x):
        """construct method"""
        original_type = x.dtype
        mean = self.mean(self.cast(x, self.compute_type), -1, keepdim=True)
        diff = self.sub(self.cast(x, self.compute_type), mean)
        varaince = self.mean2(self.square(diff), -1, keepdim=True)
        variance_eps = self.sqrt(self.add(varaince, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        output = self.cast(output, original_type)
        return output

    def shard(self, config: TransformerConfig, in_strategy=None):
        """shard method"""
        dp, _, cp = get_strategy(config)
        if in_strategy:
            strategy = in_strategy
        else:
            strategy = (cp, dp, 1)

        self.mean.shard((strategy,))
        self.sub.shard((strategy, strategy[:-1] + (1,)))
        self.square.shard((strategy,))
        self.mean2.shard((strategy,))
        self.add.shard((strategy[:-1] + (1,), ()))
        self.sqrt.shard((strategy[:-1] + (1,),))
        self.real_div.shard((strategy, strategy[:-1] + (1,)))
        self.mul.shard((strategy, (strategy[-1],)))
        self.add2.shard((strategy, (strategy[-1],)))

    def sharding_propagation(self, config: TransformerConfig):
        pass


class FusedLayerNorm(nn.Cell):
    r"""
    Layer norm operation.

    Args:
        normalized_shape (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, config, dim, eps=1e-5, param_init_type=mstype.float32, layernorm_compute_type=mstype.float32):
        super(FusedLayerNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'param_init_type' should be in [float32, float16, bfloat16], "
                            "but got the type : {}.".format(type(param_init_type)))
        if layernorm_compute_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'layernorm_compute_type' should be in [float32, float16, bfloat16], "
                            "but got the type : {}.".format(type(layernorm_compute_type)))

        self.layer_norm = P.LayerNorm(begin_norm_axis=-1,
                                      begin_params_axis=-1,
                                      epsilon=eps)
        self.gamma = Parameter(initializer('ones', dim, param_init_type), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', dim, param_init_type), name="beta",
                              parallel_optimizer=False)
        self.compute_type = layernorm_compute_type
        self.cast = P.Cast()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        else:
            self.shard(config)

    def construct(self, x):
        """construct method"""
        original_type = x.dtype
        output, _, _ = self.layer_norm(self.cast(x, self.compute_type), self.gamma, self.beta)
        output = self.cast(output, original_type)
        return output

    def shard(self, config: TransformerConfig, in_strategy=None):
        """shard method"""
        dp, _, cp = get_strategy(config)
        if in_strategy:
            strategy = in_strategy
        else:
            strategy = (cp, dp, 1)

        if strategy[-1] != 1:
            raise TypeError(
                'The last dim in FusedlayerNorm can not equal to 1! Strategy {} not supported!'.format(strategy))

        self.layer_norm.shard((strategy, (strategy[-1],), (strategy[-1],)))

    def sharding_propagation(self, config: TransformerConfig):
        pass


class RMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

    Args:
        dim (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, config, dim, eps=1e-6, param_init_type=mstype.float32, layernorm_compute_type=mstype.float32):
        super(RMSNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'param_init_type' should be in [float32, float16, bfloat16], "
                            "but got the type : {}.".format(type(param_init_type)))
        if layernorm_compute_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'layernorm_compute_type' should be in [float32, float16, bfloat16], "
                            "but got the type : {}.".format(type(layernorm_compute_type)))

        self.eps = eps
        self.weight = Parameter(initializer('ones', (dim), param_init_type))
        self.compute_type = layernorm_compute_type

        self.square = P.Square()
        self.mean = MeanExt()
        self.add = AddExt()
        self.rsqrt = Rsqrt()
        self.mul = Mul()
        self.mul2 = Mul()
        self.cast = Cast()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        else:
            self.shard(config)

    def construct(self, x):
        """construct method"""
        original_type = x.dtype
        norm_factor = self.square(self.cast(x, self.compute_type))
        norm_factor = self.mean(norm_factor, -1, keepdim=True)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        output = self.mul(self.cast(x, self.compute_type), norm_factor)
        output = self.mul2(output, self.weight)
        output = self.cast(output, original_type)
        return output

    def shard(self, config: TransformerConfig, in_strategy=None):
        """shard method"""
        dp, _, cp = get_strategy(config)
        if in_strategy:
            strategy = in_strategy
        else:
            strategy = (cp, dp, 1)

        self.square.shard((strategy,))
        self.mean.shard((strategy,))
        self.add.shard((strategy[:-1] + (1,), ()))
        self.rsqrt.shard((strategy[:-1] + (1,),))
        self.mul.shard((strategy, strategy[:-1] + (1,)))
        self.mul2.shard((strategy, (strategy[-1],)))

    def sharding_propagation(self, config: TransformerConfig):
        pass


class FusedRMSNorm(nn.Cell):
    r"""
    FusedRMSNorm operation

    Args:
        dim (tuple): The shape of the input tensor
        eps (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(self, config, dim, eps=1e-6, param_init_type=mstype.float32, layernorm_compute_type=mstype.float32):
        super(FusedRMSNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'param_init_type' should be in [float32, float16, bfloat16], "
                            "but got the type : {}.".format(type(param_init_type)))
        if layernorm_compute_type not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'layernorm_compute_type' should be in [float32, float16, bfloat16], "
                            "but got the type : {}.".format(type(layernorm_compute_type)))

        self.eps = eps
        self.weight = Parameter(initializer('ones', (dim), param_init_type))
        self.compute_type = layernorm_compute_type

        self.norm = P.RmsNorm(eps)
        self.cast = Cast()

        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(config)
        else:
            self.shard(config)

    def construct(self, x):
        """construct method"""
        original_type = x.dtype
        output = self.norm(self.cast(x, self.compute_type), self.weight)[0]
        output = self.cast(output, original_type)
        return output

    def shard(self, config: TransformerConfig, in_strategy=None):
        """shard method"""
        dp, _, cp = get_strategy(config)
        if in_strategy:
            strategy = in_strategy
        else:
            strategy = (cp, dp, 1)

        if strategy[-1] != 1 and ms.get_context('mode') == ms.GRAPH_MODE:
            raise TypeError(
                'The last dim in FusedlayerNorm can not equal to 1! Strategy {} not supported!'.format(strategy))

        self.norm.shard((strategy, (strategy[-1],)))

    def sharding_propagation(self, config: TransformerConfig):
        pass


class Norm:
    """
    Factory class for creating standard normalization layers.

    This class returns an instance of a normalization layer based on the
    `normalization` field in the provided `config`.

    Currently supported:
        - LayerNorm
        - RMSNorm

    Args:
        config (TransformerConfig): Configuration object containing model settings,
                                     including the `normalization` type.
        dim (int): The dimension of the input tensor to normalize.
        eps (float, optional): A small value to avoid division by zero. Default: 1e-5.
        param_init_type (dtype, optional): Data type for parameter initialization. Default: mstype.float32.
        layernorm_compute_type (dtype, optional): Data type for layer normalization computation.
                                                  Default: mstype.float32.

    Returns:
        A `LayerNorm` or `RMSNorm` instance.

    Raises:
        Exception: If an unsupported normalization type is specified.
    """

    @staticmethod
    def __new__(cls, config: TransformerConfig, dim, eps=1e-5, param_init_type=mstype.float32,
                layernorm_compute_type=mstype.float32):
        if config.normalization == "LayerNorm":
            return LayerNorm(
                config,
                dim=dim,
                eps=eps,
                param_init_type=param_init_type,
                layernorm_compute_type=layernorm_compute_type)
        if config.normalization == "RMSNorm":
            return RMSNorm(
                config,
                dim=dim,
                eps=eps,
                param_init_type=param_init_type,
                layernorm_compute_type=layernorm_compute_type)
        raise Exception('Only LayerNorm and RMSNorm are currently supported')


class FusedNorm:
    """
    Factory class for creating fused normalization layers.

    This class returns an instance of a fused normalization layer based on the
    `normalization` field in the provided `config`.

    Fused layers may offer better performance by combining operations for optimization.

    Currently supported:
        - FusedLayerNorm
        - FusedRMSNorm

    Args:
        config (TransformerConfig): Configuration object containing model settings,
                                     including the `normalization` type.
        dim (int): The dimension of the input tensor to normalize.
        eps (float, optional): A small value to avoid division by zero. Default: 1e-5.
        param_init_type (dtype, optional): Data type for parameter initialization. Default: mstype.float32.
        layernorm_compute_type (dtype, optional): Data type for layer normalization computation.
                                                  Default: mstype.float32.

    Returns:
        A `FusedLayerNorm` or `FusedRMSNorm` instance.

    Raises:
        Exception: If an unsupported normalization type is specified.
    """

    @staticmethod
    def __new__(cls, config: TransformerConfig, dim, eps=1e-5, param_init_type=mstype.float32,
                layernorm_compute_type=mstype.float32):
        if config.normalization == "LayerNorm":
            return FusedLayerNorm(
                config,
                dim=dim,
                eps=eps,
                param_init_type=param_init_type,
                layernorm_compute_type=layernorm_compute_type)
        if config.normalization == "RMSNorm":
            return FusedRMSNorm(
                config,
                dim=dim,
                eps=eps,
                param_init_type=param_init_type,
                layernorm_compute_type=layernorm_compute_type)
        raise Exception('Only LayerNorm and RMSNorm are currently supported')


def get_norm(config):
    r"""
    Get normalization layer.

    Args:
        config: The config of the model.

    Returns:
        callable, the normalization layer.
    """
    if config.normalization == "LayerNorm":
        return LayerNorm(
            config=config,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            param_init_type=config.params_dtype,
            layernorm_compute_type=config.layernorm_compute_type)
    if config.normalization == "FusedLayerNorm":
        return FusedLayerNorm(
            config=config,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            param_init_type=config.params_dtype,
            layernorm_compute_type=config.layernorm_compute_type)
    if config.normalization == "RMSNorm":
        return RMSNorm(
            config=config,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            param_init_type=config.params_dtype,
            layernorm_compute_type=config.layernorm_compute_type)
    if config.normalization == "FusedRMSNorm":
        return FusedRMSNorm(
            config=config,
            dim=config.hidden_size,
            eps=config.layernorm_epsilon,
            param_init_type=config.params_dtype,
            layernorm_compute_type=config.layernorm_compute_type)

    raise Exception(f"unsupported norm type '{config.normalization}'.")
