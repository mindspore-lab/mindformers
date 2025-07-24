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
"""Activation functions for transformer."""
__all__ = ["GELU", "SwiGlu", "SiLU", "get_activation"]

from mindspore import nn, dtype, Tensor
from mindspore.ops import operations as P
from mindspore.ops.auto_generate import Mul, AddExt, GeLU, Erf, Sqrt, Div, Cast
from mindspore.ops.auto_generate import SiLU as SiLU_op
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindformers.parallel_core.model_parallel_config import ModelParallelConfig


class SwiGlu(nn.Cell):
    """
    SwiGlu activation function.

    Args:
        config (ModelParallelConfig): The model parallel configuration.
    """

    def __init__(self, config: ModelParallelConfig = None):
        super(SwiGlu, self).__init__()
        self.slice = P.StridedSlice()
        # pylint: disable=W0212
        self.silu = SiLU_op()
        self.mul = Mul()
        if config is not None:
            if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
                self.sharding_propagation(config)
            else:
                self.shard(config)

    def construct(self, x: Tensor) -> Tensor:
        seq_len, bs, hidden_size = x.shape
        x_1 = self.slice(x, (0, 0, 0), (seq_len, bs, hidden_size // 2), (1, 1, 1))
        x_2 = self.slice(x, (0, 0, hidden_size // 2), (seq_len, bs, hidden_size), (1, 1, 1))
        return self.mul(self.silu(x_1), x_2)

    def shard(self, config: ModelParallelConfig):
        """
        Shard operators in GELU activation function.

        Args:
            config (ModelParallelConfig): The model parallel configuration.
        """
        dp = config.data_parallel_size if config and config.data_parallel_size is not None else 1
        cp = config.context_parallel_size if config and config.context_parallel_size is not None else 1

        silu_in_strategy = ((cp, dp, 1),)
        self.silu.shard(silu_in_strategy)
        slice_in_strategy = ((cp, dp, 1),)
        self.slice.shard(slice_in_strategy)
        mul_in_strategy = ((cp, dp, 1), (cp, dp, 1))
        self.mul.shard(mul_in_strategy)

    def sharding_propagation(self, config: ModelParallelConfig):
        pass


class GELU(nn.Cell):
    """
    GELU activation function.

    Args:
        config (ModelParallelConfig): The model parallel configuration.
        approximate (bool): Whether to use the approximate version of GELU.
    """

    def __init__(self, config: ModelParallelConfig = None, approximate: bool = True):
        super(GELU, self).__init__()
        self.approximate = approximate
        if self.approximate:
            self.gelu = GeLU()
        else:
            self.erf = Erf()
            self.sqrt = Sqrt()
            self.mul_tensor = Mul()
            self.mul_const = Mul()
            self.add_erf = AddExt()
            self.div = Div()
            self.const0 = Tensor(0.5, dtype.float32)
            self.const1 = Tensor(1.0, dtype.float32)
            self.const2 = Tensor(2.0, dtype.float32)
            self.cast = Cast()

        if config is not None:
            if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
                self.sharding_propagation(config)
            else:
                self.shard(config)

    def construct(self, x: Tensor) -> Tensor:
        """
        Apply GELU activation function.

        Args:
            x (Tensor): The input tensor.
        """

        if self.approximate:
            return self.gelu(x)

        const0 = self.cast(self.const0, x.dtype)
        const1 = self.cast(self.const1, x.dtype)
        const2 = self.cast(self.const2, x.dtype)
        return self.mul_tensor(self.mul_const(x, const0),
                               self.add_erf(const1, self.erf(self.div(x, self.sqrt(const2)))))

    def shard(self, config: ModelParallelConfig):
        """
        Shard operators in GELU activation function.

        Args:
            config (ModelParallelConfig): The model parallel configuration.
        """
        dp = config.data_parallel_size if config and config.data_parallel_size is not None else 1
        cp = config.context_parallel_size if config and config.context_parallel_size is not None else 1
        tp = config.tensor_model_parallel_size if config and config.tensor_model_parallel_size is not None else 1
        if self.approximate:
            gelu_in_strategy = ((cp, dp, tp),)
            self.gelu.shard(gelu_in_strategy)
        else:
            div_in_strategy = ((cp, dp, tp), ())
            self.div.shard(div_in_strategy)
            erf_in_strategy = ((cp, dp, tp), ())
            self.erf.shard(erf_in_strategy)
            add_erf_in_strategy = ((), (cp, dp, tp))
            self.add_erf.shard(add_erf_in_strategy)
            mul_const_in_strategy = ((cp, dp, tp), ())
            self.mul_const.shard(mul_const_in_strategy)
            mul_tensor_in_strategy = ((cp, dp, tp), (cp, dp, tp))
            self.mul_tensor.shard(mul_tensor_in_strategy)

    def sharding_propagation(self, config: ModelParallelConfig):
        pass


class SiLU(nn.Cell):
    """
    SiLU activation function.

    Args:
        config (ModelParallelConfig): The model parallel configuration.
    """

    def __init__(self, config: ModelParallelConfig = None):
        super(SiLU, self).__init__()
        # pylint: disable=W0212
        self.silu = SiLU_op()
        if config is not None:
            if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
                self.sharding_propagation(config)
            else:
                self.shard(config)

    def construct(self, x: Tensor) -> Tensor:
        return self.silu(x)

    def shard(self, config: ModelParallelConfig):
        """
        Shard operators in GELU activation function.

        Args:
            config (ModelParallelConfig): The model parallel configuration.
        """
        dp = config.data_parallel_size if config and config.data_parallel_size is not None else 1
        cp = config.context_parallel_size if config and config.context_parallel_size is not None else 1
        silu_in_strategy = ((cp, dp, 1),)
        self.silu.shard(silu_in_strategy)

    def sharding_propagation(self, config: ModelParallelConfig):
        pass


ACTIVATION_MAP = {
    'gelu': GELU,
    'swiglu': SwiGlu,
    'silu': SiLU
}


def get_activation(activation_name, *args, **kwargs):
    activation_name = activation_name.lower()
    if activation_name not in ACTIVATION_MAP:
        raise NotImplementedError(f"Activation '{activation_name}' is not supported for now. \
                                  Supported activations are: {ACTIVATION_MAP}")

    # activation should be a Cell in Static
    activation = ACTIVATION_MAP[activation_name]
    return activation(*args, **kwargs)
