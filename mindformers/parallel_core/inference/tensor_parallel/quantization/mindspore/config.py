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
"""Mindspore quantization configuration."""
from typing import Dict, Any, Optional, List, Union

import mindspore

from mindformers.parallel_core.inference.tensor_parallel.layers import (LinearBase,
                                                                        VocabParallelEmbedding,
                                                                        UnquantizedLinearMethod,
                                                                        UnquantizedEmbeddingMethod)
from mindformers.parallel_core.inference.tensor_parallel.gemm_layers import GroupedLinearBase
from mindformers.parallel_core.inference.tensor_parallel.quantization import QuantizationConfig, QuantizationBackends
from mindformers.parallel_core.inference.tensor_parallel.quantization.mindspore.a8dynw8 import A8W8DynamicLinearMethod
from mindformers.parallel_core.inference.tensor_parallel.quantization.mindspore.a8w8 import A8W8LinearMethod
from mindformers.parallel_core.inference.tensor_parallel.quantization.base_config import QuantizeMethodBase
from mindformers.parallel_core.inference.transformer.moe.experts import GroupedMLP


class OutlierSuppressionLite:
    """Class for OSL quantization configs."""

    @staticmethod
    def get_quant_method(quant_config: QuantizationConfig,
                         layer: mindspore.nn.Cell,
                         prefix: str) -> QuantizeMethodBase:
        """Get the quantize method to use for the quantized layer.

        Args:
            layer: The layer for the quant method.
            prefix: The full name of the layer in the state dict
        Returns:
            The quantize method. None if the given layer doesn't support quant
            method.
        """
        quant_method = UnquantizedLinearMethod()
        if "embedding" in prefix and isinstance(layer, VocabParallelEmbedding):
            quant_method = UnquantizedEmbeddingMethod()
        elif "attention" in prefix and isinstance(layer, LinearBase) and "linear_kv_up_proj" not in prefix:
            quant_method = A8W8LinearMethod(quant_config)
        elif "mlp" in prefix and isinstance(layer, Union[GroupedLinearBase, LinearBase, GroupedMLP]):
            quant_method = A8W8DynamicLinearMethod(quant_config)
        return quant_method

QUANTIZATION_METHOD_MAPPING = {
    "osl": OutlierSuppressionLite.get_quant_method
}


class MindSporeConfig(QuantizationConfig):
    """Class for Mindspore quantization configs."""

    def __init__(self, full_config: Dict[str, Any]) -> None:
        super().__init__()
        self.full_config = full_config
        self.quantization = full_config["quantization"]
        # osl method need config source == golden-stick
        self.is_modelslim = full_config.get("source", "modelslim") != "golden-stick"

    def get_name(self) -> QuantizationBackends:
        return "mindspore"

    def get_supported_act_dtypes(self) -> List[str]:
        return [mindspore.dtype.float16, mindspore.dtype.int8]

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_description.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantizationConfig":
        return cls(config)

    @classmethod
    def get_min_capability(cls) -> int:
        pass

    def get_quant_method(self, layer: mindspore.nn.Cell, prefix: str) -> Optional[QuantizeMethodBase]:
        get_quant = QUANTIZATION_METHOD_MAPPING[self.quantization]
        if get_quant is None:
            raise ValueError(f"Unknown quantization method: {self.quantization}")
        return get_quant(self, layer, prefix)
