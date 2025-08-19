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

__all__ = ["QuantizeMethodBase", "QuantizationConfig"]

from abc import ABC, abstractmethod
from typing import List, Any, Optional, TYPE_CHECKING

import mindspore
from mindspore import nn, Tensor

if TYPE_CHECKING:
    from mindformers.parallel_core.inference.tensor_parallel.quantization import QuantizationBackends
else:
    QuantizationBackends = str


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


class QuantizationConfig(ABC):
    """Base class for quantization configs."""

    def __init__(self):
        super().__init__()
        # mapping is updated by models as they initialize
        self.packed_modules_mapping: dict[str, list[str]] = dict()

    @abstractmethod
    def get_name(self) -> QuantizationBackends:
        """Name of the quantization method."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> List[str]:
        """List of supported activation dtypes."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """Minimum capability to support the quantization method.

        This requirement is due to the custom kernels used by the
        quantization method.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> list[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantizationConfig":
        """Create a config class from the model's quantization config."""
        raise NotImplementedError

    @staticmethod
    def get_from_keys(config: dict[str, Any], keys: list[str]) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(f"Cannot find any of {keys} in the model's "
                         "quantization config.")

    @staticmethod
    def get_from_keys_or(config: dict[str, Any], keys: list[str],
                         default: Any) -> Any:
        """Get an optional value from the model's quantization config."""
        try:
            return QuantizationConfig.get_from_keys(config, keys)
        except ValueError:
            return default

    @abstractmethod
    def get_quant_method(self, layer: mindspore.nn.Cell,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        """Get the quantize method to use for the quantized layer.

        Args:
            layer: The layer for the quant method.
            prefix: The full name of the layer in the state dict
        Returns:
            The quantize method. None if the given layer doesn't support quant
            method.
        """
        raise NotImplementedError
