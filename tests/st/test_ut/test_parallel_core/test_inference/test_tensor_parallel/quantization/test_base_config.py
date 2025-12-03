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
"""Unit tests for base_config.py"""

from typing import List, Any, Optional
import pytest
import mindspore
from mindspore import nn, Tensor

from mindformers.parallel_core.inference.quantization.base_config import (
    QuantizeMethodBase,
    QuantizationConfig,
)


class ConcreteQuantizeMethod(QuantizeMethodBase):
    """Concrete implementation of QuantizeMethodBase for testing."""

    def __init__(self):
        super().__init__()
        self.weights_created = False

    def create_weights(self, layer: nn.Cell, *_weight_args, **_extra_weight_attrs):
        """Create weights for a layer."""
        self.weights_created = True
        _ = layer

    def apply(self, layer: nn.Cell, *_args, **_kwargs) -> Tensor:
        """Apply the weights in layer to the input tensor."""
        _ = layer
        if not self.weights_created:
            raise RuntimeError("Weights must be created before applying")
        return Tensor([1.0])


class ConcreteQuantizationConfig(QuantizationConfig):
    """Concrete implementation of QuantizationConfig for testing."""

    def __init__(self, name: str = "test_quant"):
        super().__init__()
        self._name = name

    def get_name(self) -> str:
        """Name of the quantization method."""
        return self._name

    def get_supported_act_dtypes(self) -> List[str]:
        """List of supported activation dtypes."""
        return ["float16", "float32"]

    @classmethod
    def get_min_capability(cls) -> int:
        """Minimum capability to support the quantization method."""
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        """List of filenames to search for in the model directory."""
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ConcreteQuantizationConfig":
        """Create a config class from the model's quantization config."""
        name = config.get("quantization_type", "test_quant")
        return cls(name=name)

    def get_quant_method(
        self, layer: mindspore.nn.Cell, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """Get the quantize method to use for the quantized layer."""
        _ = prefix
        if isinstance(layer, nn.Dense):
            return ConcreteQuantizeMethod()
        return None


class TestQuantizeMethodBase:
    """Test class for QuantizeMethodBase."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_concrete_quantize_method_create_weights(self):
        """Test that concrete implementation can create weights."""
        method = ConcreteQuantizeMethod()
        layer = nn.Dense(10, 20)
        method.create_weights(layer)
        assert method.weights_created is True

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_concrete_quantize_method_apply(self):
        """Test that concrete implementation can apply weights."""
        method = ConcreteQuantizeMethod()
        layer = nn.Dense(10, 20)
        method.create_weights(layer)
        result = method.apply(layer)
        assert isinstance(result, Tensor)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_quantize_method_base_embedding_raises_runtime_error(self):
        """Test that embedding method raises RuntimeError by default."""
        method = ConcreteQuantizeMethod()
        layer = nn.Dense(10, 20)
        with pytest.raises(RuntimeError):
            method.embedding(layer)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_quantize_method_base_process_weights_after_loading(self):
        """Test that process_weights_after_loading returns None by default."""
        method = ConcreteQuantizeMethod()
        layer = nn.Dense(10, 20)
        method.process_weights_after_loading(layer)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_apply_without_create_weights_raises_error(self):
        """Test that apply without create_weights raises error."""
        method = ConcreteQuantizeMethod()
        layer = nn.Dense(10, 20)
        # The concrete implementation should check if weights are created
        with pytest.raises(RuntimeError):
            method.apply(layer)


class TestQuantizationConfig:
    """Test class for QuantizationConfig."""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_concrete_config_initialization(self):
        """Test that concrete config initializes packed_modules_mapping."""
        config = ConcreteQuantizationConfig()
        assert isinstance(config.packed_modules_mapping, dict)
        assert len(config.packed_modules_mapping) == 0

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_concrete_config_get_name(self):
        """Test get_name method."""
        config = ConcreteQuantizationConfig(name="test_quantization")
        assert config.get_name() == "test_quantization"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_concrete_config_get_supported_act_dtypes(self):
        """Test get_supported_act_dtypes method."""
        config = ConcreteQuantizationConfig()
        dtypes = config.get_supported_act_dtypes()
        assert isinstance(dtypes, list)
        assert "float16" in dtypes
        assert "float32" in dtypes

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_concrete_config_get_min_capability(self):
        """Test get_min_capability class method."""
        capability = ConcreteQuantizationConfig.get_min_capability()
        assert isinstance(capability, int)
        assert capability == 70

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_concrete_config_get_config_filenames(self):
        """Test get_config_filenames static method."""
        filenames = ConcreteQuantizationConfig.get_config_filenames()
        assert isinstance(filenames, list)
        assert "quantization_config.json" in filenames

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_concrete_config_from_config(self):
        """Test from_config class method."""
        config_dict = {"quantization_type": "custom_quant"}
        config = ConcreteQuantizationConfig.from_config(config_dict)
        assert isinstance(config, ConcreteQuantizationConfig)
        assert config.get_name() == "custom_quant"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_concrete_config_get_quant_method(self):
        """Test get_quant_method method."""
        config = ConcreteQuantizationConfig()
        layer = nn.Dense(10, 20)
        quant_method = config.get_quant_method(layer, prefix="dense_layer")
        assert isinstance(quant_method, ConcreteQuantizeMethod)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_concrete_config_get_quant_method_returns_none(self):
        """Test get_quant_method returns None for unsupported layer."""
        config = ConcreteQuantizationConfig()
        layer = nn.ReLU()  # Not a Dense layer
        quant_method = config.get_quant_method(layer, prefix="relu_layer")
        assert quant_method is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_from_keys_finds_first_key(self):
        """Test get_from_keys finds value using first matching key."""
        config = {"quantization_type": "test", "quant_type": "alternative"}
        result = QuantizationConfig.get_from_keys(config, ["quantization_type", "quant_type"])
        assert result == "test"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_from_keys_finds_second_key(self):
        """Test get_from_keys finds value using second key when first not present."""
        config = {"quant_type": "alternative"}
        result = QuantizationConfig.get_from_keys(config, ["quantization_type", "quant_type"])
        assert result == "alternative"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_from_keys_raises_value_error(self):
        """Test get_from_keys raises ValueError when no key found."""
        config = {"other_key": "value"}
        with pytest.raises(ValueError, match="Cannot find any of"):
            QuantizationConfig.get_from_keys(config, ["quantization_type", "quant_type"])

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_from_keys_with_empty_keys(self):
        """Test get_from_keys with empty keys list raises ValueError."""
        config = {"key": "value"}
        with pytest.raises(ValueError, match="Cannot find any of"):
            QuantizationConfig.get_from_keys(config, [])

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_from_keys_or_returns_value(self):
        """Test get_from_keys_or returns value when key exists."""
        config = {"quantization_type": "test"}
        result = QuantizationConfig.get_from_keys_or(
            config, ["quantization_type"], "default_value"
        )
        assert result == "test"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_from_keys_or_returns_default(self):
        """Test get_from_keys_or returns default when key does not exist."""
        config = {"other_key": "value"}
        default_value = "default_quant"
        result = QuantizationConfig.get_from_keys_or(
            config, ["quantization_type"], default_value
        )
        assert result == default_value

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_from_keys_or_with_none_default(self):
        """Test get_from_keys_or works with None as default."""
        config = {"other_key": "value"}
        result = QuantizationConfig.get_from_keys_or(config, ["quantization_type"], None)
        assert result is None

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_from_keys_or_with_empty_config(self):
        """Test get_from_keys_or with empty config returns default."""
        config = {}
        default_value = "default"
        result = QuantizationConfig.get_from_keys_or(config, ["any_key"], default_value)
        assert result == default_value

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_packed_modules_mapping_mutable(self):
        """Test that packed_modules_mapping can be modified."""
        config = ConcreteQuantizationConfig()
        config.packed_modules_mapping["module1"] = ["weight1", "weight2"]
        assert config.packed_modules_mapping["module1"] == ["weight1", "weight2"]
        assert len(config.packed_modules_mapping) == 1

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    def test_get_from_keys_with_different_value_types(self):
        """Test get_from_keys works with different value types."""
        config = {
            "int_value": 42,
            "float_value": 3.14,
            "list_value": [1, 2, 3],
            "dict_value": {"nested": "value"},
        }
        assert QuantizationConfig.get_from_keys(config, ["int_value"]) == 42
        assert QuantizationConfig.get_from_keys(config, ["float_value"]) == 3.14
        assert QuantizationConfig.get_from_keys(config, ["list_value"]) == [1, 2, 3]
        assert QuantizationConfig.get_from_keys(config, ["dict_value"]) == {"nested": "value"}
