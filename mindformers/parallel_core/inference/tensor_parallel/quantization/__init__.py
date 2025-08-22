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
"""quantization core module"""

__all__ = [
    "QuantizationConfig",
    "QuantizationBackends",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]

from typing import Literal, get_args

from mindformers.parallel_core.inference.tensor_parallel.quantization.base_config import QuantizationConfig


QuantizationBackends = Literal[
    "ascend",
    "golden-stick"
]

QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationBackends))

def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    """get_quantization_config"""
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    from mindformers.parallel_core.inference.tensor_parallel.quantization.mindspore_config import MindSporeConfig
    backend_to_config: dict[str, type[QuantizationConfig]] = {
        "ascend": MindSporeConfig,
        "golden-stick": MindSporeConfig,
    }

    return backend_to_config[quantization]
