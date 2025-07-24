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
"""Base class for quantization"""

__all__ = ["QuantizeMethodBase"]

from abc import ABC, abstractmethod

from mindspore import nn, Tensor


class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    @abstractmethod
    def create_weights(self, layer: nn.Cell, *weight_args, **extra_weight_attrs):
        """
        Create weights for a layer.
        The weights will be set as attributes of the layer.
        """

        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: nn.Cell, *args, **kwargs) -> Tensor:
        """
        Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer.
        """

        raise NotImplementedError

    # Not required functions
    def embedding(self, layer: nn.Cell, *args, **kwargs) -> Tensor:
        """
        Gather embeddings in the layer based on indices in the input tensor.
        Expects create_weights to have been called before on the layer.
        """

        raise RuntimeError

    def process_weights_after_loading(self, layer: nn.Cell) -> None:
        """
        Process the weight after loading.
        This can be used for example, to transpose weights for computation.
        """

        return
