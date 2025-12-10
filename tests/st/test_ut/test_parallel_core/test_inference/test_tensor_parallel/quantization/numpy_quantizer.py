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
"""NumpyQuantizer for test."""


import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from safetensors.numpy import save_file
from simple_gpt_model import ModelSpec


@dataclass
class QuantResult:
    """Result of weight quantization."""
    weights: Dict[str, np.ndarray]
    descriptions: Dict[str, str]


class WeightQuantStrategy(ABC):
    """Abstract base class for weight quantization strategies."""

    @abstractmethod
    def quantize_weight(self, weight: np.ndarray, transpose_b: bool,
                       input_size: int = None) -> Dict[str, np.ndarray]:
        """Quantize a single weight tensor."""

    @abstractmethod
    def get_description(self) -> str:
        """Get the quantization description string."""

    @staticmethod
    def get_quant_min_max(num_bits: int = 8, signed: bool = True,
                         narrow_range: bool = False) -> Tuple[int, int]:
        """Calculate quantization params for minimum/maximum quantization integer."""
        if signed:
            quant_min = -(2 ** (num_bits - 1))
            quant_max = 2 ** (num_bits - 1) - 1
        else:
            quant_min = 0
            quant_max = 2 ** num_bits - 1

        if narrow_range:
            quant_min = quant_min + 1

        return quant_min, quant_max

    @staticmethod
    def act_int8_quant(tensor: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Quantize activation tensor to int8."""
        bits = 8
        quant_min, quant_max = WeightQuantStrategy.get_quant_min_max(bits)

        min_val = np.min(tensor)
        max_val = np.max(tensor)

        if (max_val == min_val).all():
            scale = np.array([1.0], dtype=np.float32)
            zero_point = np.array([0.0], dtype=np.float32)
        else:
            min_val = min_val.astype(np.float64)
            max_val = max_val.astype(np.float64)
            scale = (max_val - min_val) / (quant_max - quant_min)
            zero_point = quant_min - min_val / scale.astype(np.float32)
            scale = scale.astype(np.float32)

        quantized = np.round(tensor / scale + zero_point)
        quantized = np.clip(quantized, quant_min, quant_max).astype(np.int8)

        return quantized, scale, zero_point

    @staticmethod
    def weight_int8_quant(tensor: np.ndarray,
                         transpose_b: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize weight tensor to int8."""
        bits = 8
        quant_min, quant_max = WeightQuantStrategy.get_quant_min_max(bits)
        oc_axis = 0 if transpose_b else 1
        ic_axis = 1 if transpose_b else 0
        oc = tensor.shape[oc_axis]

        min_val = np.min(tensor, axis=ic_axis, keepdims=True)
        max_val = np.max(tensor, axis=ic_axis, keepdims=True)

        if (max_val == min_val).all():
            scale = np.ones((oc,), dtype=np.float32)
        else:
            min_val = min_val.astype(np.float64)
            max_val = max_val.astype(np.float64)
            max_val = np.maximum(np.abs(min_val), np.abs(max_val))
            min_val = -max_val
            scale = ((max_val - min_val) / (quant_max - quant_min)).astype(np.float32)

        quantized = np.round(tensor / scale)
        quantized = np.clip(quantized, quant_min, quant_max).astype(np.int8)
        scale = np.squeeze(scale)

        return quantized, scale

    @staticmethod
    def weight_int4_per_group_pack(tensor: np.ndarray, group_size: int,
                                   transpose_b: bool = True) -> (
                                       Tuple[np.ndarray, np.ndarray]
                                   ):
        """
        Quantize weight tensor to int4 per group and pack two int4 values into one uint8.

        Args:
            tensor: weight tensor to quantize, shape (oc, ic) if transpose_b=True
            group_size: size of each quantization group along input dimension
            transpose_b: whether the tensor is in (oc, ic) format

        Returns:
            packed: packed int4 weights as uint8, shape (oc//2, ic)
            scale_uint64: quantization scales as uint64, shape (oc, ic//group_size)
        """
        bits = 4
        quant_min, quant_max = WeightQuantStrategy.get_quant_min_max(bits, signed=True)

        if transpose_b:
            oc, ic = tensor.shape[0], tensor.shape[1]
        else:
            ic, oc = tensor.shape[0], tensor.shape[1]

        # Validate dimensions
        if ic % group_size != 0:
            raise ValueError(
                f"Input dimension {ic} must be divisible by group_size {group_size}"
            )
        if oc % 2 != 0:
            raise ValueError(
                f"Output dimension {oc} must be even for int4 packing"
            )

        num_groups = ic // group_size

        # Reshape tensor for per-group quantization: (oc, num_groups, group_size)
        if transpose_b:
            tensor_grouped = tensor.reshape(oc, num_groups, group_size)
        else:
            tensor_grouped = tensor.T.reshape(oc, num_groups, group_size)

        # Calculate scale per group (symmetric quantization)
        max_vals = np.max(np.abs(tensor_grouped), axis=2, keepdims=True)
        max_vals = np.where(max_vals == 0, 1.0, max_vals)
        scales = (max_vals / quant_max).astype(np.float32)

        # Quantize and reshape
        quantized = np.round(tensor_grouped / scales)
        quantized = np.clip(quantized, quant_min, quant_max).astype(np.int8)
        quantized = quantized.reshape(oc, ic)
        scales = scales.squeeze(axis=2)

        # Pack two consecutive oc values into one uint8
        quantized_even = quantized[0::2, :]
        quantized_odd = quantized[1::2, :]

        even_unsigned = (quantized_even & 0x0F).astype(np.uint8)
        odd_unsigned = (quantized_odd & 0x0F).astype(np.uint8)

        # Pack: even in low 4 bits, odd in high 4 bits
        packed_unsigned = (odd_unsigned << 4) | even_unsigned

        return (packed_unsigned,
                scales.astype(np.float32).view(np.uint32).astype(np.uint64))


class A8W8Strategy(WeightQuantStrategy):
    """INT8 weight and activation quantization strategy."""

    def __init__(self, quant_input: np.ndarray):
        self.quant_input = quant_input

    def quantize_weight(self, weight: np.ndarray, transpose_b: bool,
                       input_size: int = None) -> Dict[str, np.ndarray]:
        """Quantize weight using INT8 static quantization."""
        _, input_scale, input_offset = self.act_int8_quant(self.quant_input)
        quant_weight, w_scale = self.weight_int8_quant(weight, transpose_b)

        x_zp = input_offset.astype(np.int32)
        quant_bias = -np.sum(x_zp * quant_weight.astype(np.int32), axis=-1).astype(np.int32)
        deq_scale = input_scale.astype(np.float32) * w_scale.astype(np.float32)
        output_size = weight.shape[0]
        beta = np.zeros(output_size, dtype=np.int32)

        # Input scale and offset should match input_size
        input_scale_arr = np.full((input_size,), input_scale, dtype=np.float32)
        input_offset_arr = np.full((input_size,), input_offset, dtype=np.float32)

        return {
            'weight': quant_weight,
            'deq_scale': deq_scale,
            'input_scale': input_scale_arr,
            'input_offset': input_offset_arr.astype(np.int8),
            'quant_bias': quant_bias,
            'beta': beta,
        }

    def get_description(self) -> str:
        return "W8A8"


class A8DynW8Strategy(WeightQuantStrategy):
    """INT8 dynamic weight quantization strategy."""

    def quantize_weight(self, weight: np.ndarray, transpose_b: bool,
                       input_size: int = None) -> Dict[str, np.ndarray]:
        """Quantize weight using INT8 dynamic quantization."""
        quant_weight, w_scale = self.weight_int8_quant(weight, transpose_b)
        return {
            'weight': quant_weight,
            'w_scale': w_scale,
        }

    def get_description(self) -> str:
        return "W8A8_DYNAMIC"


class A8W4Strategy(WeightQuantStrategy):
    """INT4 weight quantization strategy."""

    def __init__(self, group_size: int = 256):
        self.group_size = group_size

    def quantize_weight(self, weight: np.ndarray, transpose_b: bool,
                       input_size: int = None) -> Dict[str, np.ndarray]:
        """Quantize weight using INT4 per-group quantization."""
        qweight_packed, w_scale = self.weight_int4_per_group_pack(
            weight, self.group_size, transpose_b
        )
        return {
            'weight': qweight_packed,
            'w_scale': w_scale,
        }

    def get_description(self) -> str:
        return "W4A8_DYNAMIC"


class FloatStrategy(WeightQuantStrategy):
    """No quantization (float) strategy."""

    def quantize_weight(self, weight: np.ndarray, transpose_b: bool,
                       input_size: int = None) -> Dict[str, np.ndarray]:
        """Return weight as-is without quantization."""
        return {'weight': weight}

    def get_description(self) -> str:
        return "FLOAT"


class LayerWeightHandler:
    """Handler for processing weights of different layer types."""

    def __init__(self, index: int, linear_spec, weights: dict, strategy: WeightQuantStrategy):
        self.index = index
        self.linear_spec = linear_spec
        self.weights = weights
        self.strategy = strategy

    def process(self) -> QuantResult:
        """Process weights based on layer type."""
        layer_type = self.linear_spec.linear_type

        if layer_type == "QKVParallelLinear":
            return self._process_qkv()
        if layer_type == "MergedColumnParallelLinear":
            return self._process_merged()
        if layer_type in ("ColumnParallelGroupedLinear", "RowParallelGroupedLinear"):
            return self._process_grouped()
        return self._process_standard()

    def _process_qkv(self) -> QuantResult:
        """Process QKV parallel linear weights."""
        quant_weights = {}
        quant_desc = {}

        for qkv_name in ['q', 'k', 'v']:
            weight_key = f"linears.{self.index}.{qkv_name}.weight"
            weight = self.weights[weight_key]

            quant_result = self.strategy.quantize_weight(
                weight, self.linear_spec.transpose_b, self.linear_spec.input_size
            )

            # Add quantized weights with proper keys
            for suffix, value in quant_result.items():
                key = f"linears.{self.index}.{qkv_name}.{suffix}"
                quant_weights[key] = value
                quant_desc[key] = self.strategy.get_description()

            # Add bias if present
            if self.linear_spec.has_bias:
                bias_key = f"linears.{self.index}.{qkv_name}.bias"
                quant_weights[bias_key] = self.weights[bias_key]
                quant_desc[bias_key] = self.strategy.get_description()

        return QuantResult(quant_weights, quant_desc)

    def _process_merged(self) -> QuantResult:
        """Process merged column parallel linear weights."""
        quant_weights = {}
        quant_desc = {}

        for merge_name in ['gating', 'hidden']:
            weight_key = f"linears.{self.index}.{merge_name}.weight"
            weight = self.weights[weight_key]

            quant_result = self.strategy.quantize_weight(
                weight, self.linear_spec.transpose_b, self.linear_spec.input_size
            )

            # Add quantized weights with proper keys
            for suffix, value in quant_result.items():
                key = f"linears.{self.index}.{merge_name}.{suffix}"
                quant_weights[key] = value
                quant_desc[key] = self.strategy.get_description()

            # Add bias if present
            if self.linear_spec.has_bias:
                bias_key = f"linears.{self.index}.{merge_name}.bias"
                quant_weights[bias_key] = self.weights[bias_key]
                quant_desc[bias_key] = self.strategy.get_description()

        return QuantResult(quant_weights, quant_desc)

    def _process_grouped(self) -> QuantResult:
        """Process grouped linear (MoE) weights."""
        quant_weights = {}
        quant_desc = {}

        for gate_name in ['gate', 'up']:
            weight_key = f"linears.{self.index}.{gate_name}.weight"
            weight = self.weights[weight_key]

            quant_result = self.strategy.quantize_weight(weight, transpose_b=True)

            # Add quantized weights with proper keys
            for suffix, value in quant_result.items():
                key = f"linears.{self.index}.{gate_name}.{suffix}"
                quant_weights[key] = value

        # Description uses base key for grouped layers
        quant_desc[f"linears.{self.index}.weight"] = self.strategy.get_description()
        quant_desc[f"linears.{self.index}.w_scale"] = self.strategy.get_description()

        return QuantResult(quant_weights, quant_desc)

    def _process_standard(self) -> QuantResult:
        """Process standard linear layer weights."""
        quant_weights = {}
        quant_desc = {}

        weight_key = f"linears.{self.index}.weight"
        weight = self.weights[weight_key]

        quant_result = self.strategy.quantize_weight(
            weight, self.linear_spec.transpose_b, self.linear_spec.input_size
        )

        # Add quantized weights with proper keys
        for suffix, value in quant_result.items():
            key = f"linears.{self.index}.{suffix}"
            quant_weights[key] = value
            quant_desc[key] = self.strategy.get_description()

        # Add bias if present
        if self.linear_spec.has_bias:
            bias_key = f"linears.{self.index}.bias"
            quant_weights[bias_key] = self.weights[bias_key]
            quant_desc[bias_key] = self.strategy.get_description()

        return QuantResult(quant_weights, quant_desc)


class NumpyQuantizer:
    """A class for quantizing model weights using NumPy."""

    def __init__(self, model_spec: ModelSpec, quant_policy: list):
        self.model_spec = model_spec
        self.quant_policy = quant_policy
        self.global_group_size = None

    def quant(self, quant_input: np.ndarray, weights: dict, save_dir: str):
        """Quantize the input and weights, save to safetensors and JSON description."""
        quant_weights, quant_desc = self._quant(quant_input, weights)
        print(f"quant_weights: {quant_weights.keys()}", flush=True)
        print(f"quant_desc: {quant_desc}", flush=True)

        save_file(quant_weights, os.path.join(save_dir, 'quant-model-00001-00001.safetensors'))
        with open(os.path.join(save_dir, "quantization_description.json"), "w",
                  encoding='utf-8') as f:
            json.dump(quant_desc, f, indent=2, ensure_ascii=False)

        print(f"quantization weights saved to {save_dir}", flush=True)

    def _quant(self, quant_input: np.ndarray, weights: dict) -> Tuple[dict, dict]:
        """Internal method to perform quantization on weights based on policy."""
        all_quant_weights = {}
        all_quant_desc = {}

        for index, (qpolicy, linear_spec) in enumerate(
            zip(self.quant_policy, self.model_spec.linear_specs)
        ):
            # Create appropriate quantization strategy
            strategy = self._create_strategy(qpolicy, quant_input, linear_spec)

            # Process weights using the strategy
            handler = LayerWeightHandler(index, linear_spec, weights, strategy)
            result = handler.process()

            # Merge results
            all_quant_weights.update(result.weights)
            all_quant_desc.update(result.descriptions)

        # Add global group size if set
        if self.global_group_size is not None:
            all_quant_desc["group_size"] = int(self.global_group_size)

        return all_quant_weights, all_quant_desc

    def _create_strategy(self, qpolicy: str, quant_input: np.ndarray,
                        linear_spec) -> WeightQuantStrategy:
        """Create appropriate quantization strategy based on policy."""
        if qpolicy == 'a8w8':
            return A8W8Strategy(quant_input)
        if qpolicy == 'a8dynw8':
            return A8DynW8Strategy()
        if qpolicy == 'a8w4':
            # Validate that a8w4 is only used with grouped layers
            layer_type = linear_spec.linear_type
            if layer_type not in ("ColumnParallelGroupedLinear",
                                  "RowParallelGroupedLinear"):
                raise ValueError(
                    "a8w4 quantization only supports grouped linear layers"
                )
            group_size = 256
            self.global_group_size = group_size
            return A8W4Strategy(group_size)
        if qpolicy is None or qpolicy == 'float':
            return FloatStrategy()

        raise ValueError(f"Unsupported quant policy: {qpolicy}")
