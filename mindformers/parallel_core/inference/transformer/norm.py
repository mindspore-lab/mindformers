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
"""Normalization"""
__all__ = ["get_norm_cls"]

import mindspore.common.dtype as mstype
import mindspore.ops.operations as P
from mindspore import Parameter, nn
from mindspore.common.initializer import initializer

from mindformers.parallel_core.transformer_config import TransformerConfig


class LayerNorm(nn.Cell):
    r"""
    Layer norm operation.

    Args:
        config: Transformer Config.
        hidden_size (tuple): The shape of the parameter.
        eps (float): The epsilon value of the denominator. Default 1e-5.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch, seq_length, hidden_size).

    Outputs:
        - Tensor with shape (batch, seq_length, hidden_size).
    """

    def __init__(
            self,
            config: TransformerConfig,
            hidden_size: int,
            eps: float = 1e-5,
    ):
        super().__init__()
        if config.layernorm_compute_dtype not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'layernorm_compute_dtype' should in [float32, float16, bfloat16], "
                            "but got the type : {}.".format(type(config.layernorm_compute_dtype)))

        self.compute_type = config.layernorm_compute_dtype
        self.hidden_size = hidden_size

        self.gamma = Parameter(initializer('ones', hidden_size, self.compute_type), name="gamma",
                               parallel_optimizer=False, requires_grad=False)
        self.beta = Parameter(initializer('zeros', hidden_size, self.compute_type), name="beta",
                              parallel_optimizer=False, requires_grad=False)

        self.layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=eps)
        self.cast = P.Cast()

    def construct(self, x):
        """construct method"""
        original_type = x.dtype
        x = self.cast(x, self.compute_type)
        output, _, _ = self.layer_norm(x, self.gamma, self.beta)
        output = self.cast(output, original_type)
        return output


class RMSNorm(nn.Cell):
    r"""
    A self-defined RMSNorm operation using reduce mean.

    Args:
        config: Transformer Config.
        hidden_size (tuple): The shape of the parameter.
        eps (float): The epsilon value of the denominator. Default 1e-5.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch, seq_length, hidden_size)`.

    Outputs:
        Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(
            self,
            config: TransformerConfig,
            hidden_size: int,
            eps: float = 1e-5,
    ):
        super().__init__()
        if config.layernorm_compute_dtype not in [mstype.float32, mstype.float16, mstype.bfloat16]:
            raise TypeError("The type of parameter 'layernorm_compute_dtype' should in [float32, float16, bfloat16], "
                            "but got the type : {}.".format(type(config.layernorm_compute_dtype)))

        self.compute_type = config.layernorm_compute_dtype
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = Parameter(initializer('ones', (hidden_size,), dtype=self.compute_type), name="weight",
                                parallel_optimizer=False, requires_grad=False)

        self.cast = P.Cast()
        self.norm = P.RmsNorm(eps)

    def construct(self, x):
        """Forward of RMSNorm."""
        original_type = x.dtype
        output = self.norm(self.cast(x, self.compute_type), self.weight)[0]
        output = self.cast(output, original_type)
        return output

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (1,)
        state_dict = {}
        state_dict[self.weight.name] = {'shape': self.weight.shape,
                                        'shard': w_shard}
        return state_dict


def get_norm_cls(normalization: str):
    r"""
    Get the class of normalization layer.

    Args:
        normalization (str): The normalization type.

    Returns:
        callable, the class of normalization layer.
    """
    norm_map = {
        "LayerNorm": LayerNorm,
        "RMSNorm": RMSNorm,
    }
    if normalization not in norm_map.keys():
        raise Exception(f"unsupported norm type '{normalization}'.")
    return norm_map[normalization]
