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
import numpy as np
from safetensors.numpy import save_file
from gpt_model_for_test import ModelSpec


class NumpyQuantizer:
    """A class for quantizing model weights using NumPy."""

    def __init__(self, model_spec: ModelSpec, quant_policy: list[str]):
        self.model_spec = model_spec
        self.quant_policy = quant_policy
        self.description_file_path = None

    def quant(self, quant_input: np.ndarray, weights, save_dir):
        """Quantize the input and weights, save to safetensors and JSON description."""
        quant_weights, quant_desc = self._quant(quant_input, weights)
        print(f"quant_weights: {quant_weights.keys()}", flush=True)
        print(f"quant_desc: {quant_desc}", flush=True)
        save_file(quant_weights, os.path.join(save_dir, 'quant-model-00001-00001.safetensors'))
        with open(os.path.join(save_dir, "quantization_description.json"), "w", encoding='utf-8') as f:
            json.dump(quant_desc, f, indent=2, ensure_ascii=False)
        print(f"quantization weights saved to {save_dir}", flush=True)

    def _quant(self, quant_input: np.ndarray, weights):
        """Internal method to perform quantization on weights based on policy."""
        quant_weights = {}
        quant_desc = {}
        for index, (qpolicy, linear_spec) in enumerate(zip(self.quant_policy, self.model_spec.linear_specs)):
            weight = weights[f"linears.{index}.weight"]
            if qpolicy == 'a8w8':
                _, input_scale, input_offset = NumpyQuantizer._act_int8_quant(quant_input)
                quant_weight, w_scale = NumpyQuantizer._weight_int8_quant(weight, transpose_b=linear_spec.transpose_b)
                x_zp = input_offset.astype(np.int32)  # per-tensor zero-point
                quant_bias = -np.sum(x_zp * quant_weight.astype(np.int32), axis=-1).astype(np.int32)
                deq_scale = (input_scale.astype(np.float32) * w_scale.astype(np.float32))
                beta = np.zeros(linear_spec.input_size, dtype=np.int32)
                quant_weights.update({
                        f"linears.{index}.weight": quant_weight,
                        f"linears.{index}.deq_scale": deq_scale,
                        f"linears.{index}.input_scale": np.tile(input_scale, linear_spec.input_size),
                        f"linears.{index}.input_offset": np.tile(input_offset, linear_spec.input_size),
                        f"linears.{index}.quant_bias": quant_bias,
                        f"linears.{index}.beta": beta,
                    })
                quant_desc.update({
                        f"linears.{index}.weight": "W8A8",
                        f"linears.{index}.deq_scale": "W8A8",
                        f"linears.{index}.input_scale": "W8A8",
                        f"linears.{index}.input_offset": "W8A8",
                        f"linears.{index}.quant_bias": "W8A8",
                        f"linears.{index}.beta": "W8A8",
                    })
                if linear_spec.has_bias:
                    quant_weights[f"linears.{index}.bias"] = weights[f"linears.{index}.bias"]
                    quant_desc[f"linears.{index}.bias"] = "W8A8"
                continue
            if qpolicy == 'a8dynw8':
                quant_weight, w_scale = NumpyQuantizer._weight_int8_quant(weight, transpose_b=linear_spec.transpose_b)
                quant_weights.update({
                        f"linears.{index}.weight": quant_weight,
                        f"linears.{index}.w_scale": w_scale
                    })
                quant_desc.update({
                        f"linears.{index}.weight": "W8A8_DYNAMIC",
                        f"linears.{index}.w_scale": "W8A8_DYNAMIC",
                    })
                if linear_spec.has_bias:
                    quant_weights[f"linears.{index}.bias"] = weights[f"linears.{index}.bias"]
                    quant_desc[f"linears.{index}.bias"] = "W8A8_DYNAMIC"
                continue
            if qpolicy is None:
                quant_weights.update({
                        f"linears.{index}.weight": weight,
                    })
                quant_desc.update({
                        f"linears.{index}.weight": "FLOAT",
                    })
                if linear_spec.has_bias:
                    quant_weights[f"linears.{index}.bias"] = weights[f"linears.{index}.bias"]
                    quant_desc[f"linears.{index}.bias"] = "FLOAT"
                continue
            raise ValueError(f"Unsupported quant policy: {qpolicy}")
        return quant_weights, quant_desc

    @staticmethod
    def _get_quant_min_max(num_bits=8, signed=True, narrow_range=False):
        """Calculate quantization params for minimum/maximum quantization integer"""
        if signed:
            quant_min = 0 - 2 ** (num_bits - 1)
            quant_max = 2 ** (num_bits - 1) - 1
        else:
            quant_min = 0
            quant_max = 2 ** num_bits - 1
        if narrow_range:
            quant_min = quant_min + 1
        return quant_min, quant_max

    @staticmethod
    def _act_int8_quant(tensor):
        """Quantize activation tensor to int8."""
        bits=8
        quant_min, quant_max = NumpyQuantizer._get_quant_min_max(bits)

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
    def _weight_int8_quant(tensor, transpose_b=True):
        """Quantize weight tensor to int8."""
        bits=8
        quant_min, quant_max = NumpyQuantizer._get_quant_min_max(bits)
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
