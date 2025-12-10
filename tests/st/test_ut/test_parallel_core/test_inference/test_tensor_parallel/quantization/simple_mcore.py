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
"""Simple MCore parallel linear inference runner with YAML configuration"""


import argparse
import glob
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

import mindspore as ms
from mindspore.communication import init
from numpy_quantizer import NumpyQuantizer
from simple_gpt_model import (SimpleGPTModel, LinearSpec, ModelSpec,
                               QKVLinearSpec, GroupLinearSpec, convert_dtype_str_to_ms)
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups


@dataclass
class WeightLoadingInfo:
    """Information about how to load a weight parameter."""
    cleaned_key: str
    shard_id: Optional[str] = None
    expert_id: Optional[int] = None
    layer_type: str = "standard"  # "standard", "qkv", "merged", "grouped"


class WeightKeyParser:
    """Parser for weight keys to extract loading information."""

    # Define weight patterns with their parsing rules
    WEIGHT_PATTERNS = [
        (".gate", "w1", 0, "grouped"),
        (".up", "w3", 0, "grouped"),
        (".gating", "gating", None, "merged"),
        (".hidden", "hidden", None, "merged"),
        (".q.", "q", None, "qkv"),
        (".k.", "k", None, "qkv"),
        (".v.", "v", None, "qkv"),
    ]

    @staticmethod
    def parse(weight_key: str) -> WeightLoadingInfo:
        """Parse a weight key and return loading information."""
        for pattern, shard_id, expert_id, layer_type in WeightKeyParser.WEIGHT_PATTERNS:
            if pattern in weight_key:
                cleaned_key = weight_key.replace(pattern.rstrip('.'), '')
                return WeightLoadingInfo(cleaned_key, shard_id, expert_id, layer_type)

        # Default: standard layer without special shard_id
        return WeightLoadingInfo(weight_key, None, None, "standard")


class WeightGenerator:
    """Generator for creating random weights for different layer types."""

    @staticmethod
    def generate(model_spec: ModelSpec) -> dict:
        """Generate random float weights for all layers in the model spec."""
        np.random.seed(42)
        weights = {}

        for index, linear_spec in enumerate(model_spec.linear_specs):
            layer_weights = WeightGenerator._generate_for_layer(index, linear_spec)
            weights.update(layer_weights)

        return weights

    @staticmethod
    def _generate_for_layer(index: int, linear_spec) -> dict:
        """Generate weights for a single layer based on its type."""
        layer_type = linear_spec.linear_type

        if layer_type == "QKVParallelLinear":
            return WeightGenerator._generate_qkv(index, linear_spec)
        if layer_type == "ColumnParallelGroupedLinear":
            return WeightGenerator._generate_grouped(index, linear_spec)
        if layer_type == "MergedColumnParallelLinear":
            return WeightGenerator._generate_merged(index, linear_spec)
        return WeightGenerator._generate_standard(index, linear_spec)

    @staticmethod
    def _generate_qkv(index: int, spec) -> dict:
        """Generate weights for QKV parallel linear."""
        weights = {}
        qkv_names = ["q", "k", "v"]

        for name, output_size in zip(qkv_names, spec.output_sizes):
            weight_shape = (output_size, spec.input_size)
            weights[f"linears.{index}.{name}.weight"] = (
                0.01 * np.random.randn(*weight_shape).astype(np.float32)
            )

            if spec.has_bias:
                weights[f"linears.{index}.{name}.bias"] = (
                    0.01 * np.random.randn(output_size).astype(np.float32)
                )

        return weights

    @staticmethod
    def _generate_grouped(index: int, spec) -> dict:
        """Generate weights for grouped linear (MoE)."""
        weights = {}
        half_size = spec.output_size // 2

        for name in ["gate", "up"]:
            weight_shape = (half_size, spec.input_size)
            weights[f"linears.{index}.{name}.weight"] = 0.01 * np.random.randn(*weight_shape).astype(np.float32)

        return weights

    @staticmethod
    def _generate_merged(index: int, spec) -> dict:
        """Generate weights for merged column parallel linear."""
        weights = {}
        weight_shape = (spec.output_size, spec.input_size)

        for name in ["gating", "hidden"]:
            weights[f"linears.{index}.{name}.weight"] = 0.01 * np.random.randn(*weight_shape).astype(np.float32)

            if spec.has_bias:
                weights[f"linears.{index}.{name}.bias"] = 0.01 * np.random.randn(spec.output_size).astype(np.float32)

        return weights

    @staticmethod
    def _generate_standard(index: int, spec) -> dict:
        """Generate weights for standard linear layers."""
        weights = {}
        weight_shape = (spec.output_size, spec.input_size)

        weights[f"linears.{index}.weight"] = 0.01 * np.random.randn(*weight_shape).astype(np.float32)

        if spec.has_bias:
            weights[f"linears.{index}.bias"] = 0.01 * np.random.randn(spec.output_size).astype(np.float32)

        return weights


class WeightLoader:
    """Loader for network weights with support for different layer types."""

    @staticmethod
    def load_into_network(network, weights: dict):
        """Load weights into network parameters."""
        params = network.parameters_dict()
        print(params)
        loaded = []

        for original_key, weight_value in weights.items():
            load_info = WeightKeyParser.parse(original_key)
            param = params.get(load_info.cleaned_key)

            if param is None:
                continue

            WeightLoader._load_single_weight(param, weight_value, load_info)
            loaded.append(original_key)

        # Report loading status
        WeightLoader._report_loading_status(weights, loaded, params)

    @staticmethod
    def _load_single_weight(param, weight_value, load_info: WeightLoadingInfo):
        """Load a single weight into a parameter based on its type."""
        if load_info.layer_type == "grouped":
            # ColumnParallelGroupedLinear: needs both shard_id and expert_id
            param.weight_loader(param, weight_value, load_info.shard_id, load_info.expert_id)
        elif load_info.shard_id is not None:
            # QKV or Merged layers: needs shard_id only
            param.weight_loader(param, weight_value, load_info.shard_id)
        else:
            # Standard layers: no special arguments
            param.weight_loader(param, weight_value)

    @staticmethod
    def _report_loading_status(weights: dict, loaded: list, params: dict):
        """Report which weights were not used and which params were not loaded."""
        weights_not_used = set(weights.keys()) - set(loaded)
        params_not_loaded = set(params.keys()) - set(loaded)

        if weights_not_used:
            print(f"weights not used: {weights_not_used}", flush=True)
        if params_not_loaded:
            print(f"params not loaded: {params_not_loaded}", flush=True)


class LinearSpecFactory:
    """Factory for creating LinearSpec instances based on layer type."""

    @staticmethod
    def create(layer_type: str, has_bias: bool, quant_policy: str, config) -> object:
        """Create appropriate LinearSpec based on layer type."""
        if layer_type == "QKVParallelLinear":
            return QKVLinearSpec(
                layer_type, config.input_size, config.head_size,
                config.total_num_heads, config.total_num_kv_heads,
                has_bias, config.compute_dtype, quant_policy
            )
        if layer_type == "ColumnParallelGroupedLinear":
            return GroupLinearSpec(
                layer_type, config.num_local_experts, config.input_size,
                config.output_size, quant_policy
            )
        # Standard and merged layers use LinearSpec
        return LinearSpec(
            layer_type, config.input_size, config.output_size,
            has_bias, config.compute_dtype, quant_policy
        )


class SimpleMCoreRunner:
    """Simple runner for MCore parallel linear layers with quantization support."""

    def __init__(self, config):
        """Initialize the simple MCore runner with given arguments."""
        self.config = config
        self._setup_parallel_context()
        self._load_config_and_build_model()

    def _setup_parallel_context(self):
        """Setup parallel computing context."""
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None
        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))
        self.model_comm_pgs = ModelCommProcessGroups.get_default_model_comm_pgs()

        if self.rank_id is not None:
            init()
            initialize_model_parallel(tensor_model_parallel_size=self.config.tensor_parallel)
            self.model_comm_pgs = ModelCommProcessGroups.use_parallel_state_groups(required_groups=['tp'])

    def _load_config_and_build_model(self):
        """Load YAML configuration and build model specification."""
        # Load YAML configuration
        config_path = Path(self.config.config_file)
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)

        # Extract model configuration
        model_config = yaml_config['model_config']
        self._set_model_config(model_config)

        # Build linear specs from test cases
        test_cases = yaml_config['test_cases']
        linear_specs, quant_policies = self._build_linear_specs(test_cases)

        # Create model spec
        self.model_spec = ModelSpec(
            compute_dtype=self.compute_dtype,
            param_init_dtype=self.param_init_dtype,
            tensor_parallel=self.config.tensor_parallel,
            linear_specs=linear_specs,
        )

        # Setup quantization if needed
        self.quantization = self.config.quantization
        self.quant_model_dir = None
        if self.quantization == 'golden-stick':
            self.quantizer = NumpyQuantizer(self.model_spec, quant_policies)
            self.quant_model_dir = tempfile.mkdtemp(prefix="quant_model_for_test_")

    def _set_model_config(self, model_config: dict):
        """Set model configuration from YAML."""
        self.input_size = model_config['input_size']
        self.output_size = model_config['output_size']
        self.head_size = model_config['head_size']
        self.total_num_heads = model_config['total_num_heads']
        self.total_num_kv_heads = model_config['total_num_kv_heads']
        self.compute_dtype = model_config['compute_dtype']
        self.param_init_dtype = model_config['param_init_dtype']
        self.num_local_experts = model_config['num_local_experts']

    def _build_linear_specs(self, test_cases: list) -> Tuple[list, list]:
        """Build linear specs and quant policies from test cases."""
        linear_specs = []
        quant_policies = []

        for case in test_cases:
            quant_policy = case['quant_policy']
            quant_policy = quant_policy if self.config.quantization == 'golden-stick' else 'float'

            linear_spec = LinearSpecFactory.create(
                case['linear_type'],
                case['has_bias'],
                quant_policy,
                self
            )

            linear_specs.append(linear_spec)
            quant_policies.append(quant_policy)

        return linear_specs, quant_policies

    def _create_network(self):
        """Create the network model for testing."""
        return SimpleGPTModel(self.model_spec, self.model_comm_pgs, self.quantization, self.quant_model_dir)

    def _load_quant_weights(self) -> dict:
        """Load quantized weights from the model directory."""
        if not os.path.isdir(self.quant_model_dir):
            raise ValueError(f"Invalid quant_model_dir: {self.quant_model_dir}")

        safetensor_files = glob.glob(os.path.join(self.quant_model_dir, "*.safetensors"))
        if len(safetensor_files) != 1:
            raise FileNotFoundError(
                f"Expected 1 safetensor file in {self.quant_model_dir}, found {len(safetensor_files)}"
            )

        safetensor_file = safetensor_files[0]
        with safe_open(safetensor_file, framework="np", device="cpu") as f:
            return {key: f.get_slice(key) for key in f.keys()}

    def _prepare_weights(self, weights: dict) -> dict:
        """Prepare weights for loading (convert to safetensors if needed)."""
        first_value = next(iter(weights.values()))

        # MoE must use safetensors format
        if isinstance(first_value, np.ndarray):
            with tempfile.TemporaryDirectory() as temp_dir:
                path = os.path.join(temp_dir, "model.safetensors")
                save_file(weights, path)
                weights.clear()
                with safe_open(path, framework="np", device="cpu") as f:
                    weights = {key: f.get_slice(key) for key in f.keys()}

        return weights

    def run(self):
        """Run the simple MCore test."""
        # Generate input and weights
        input_data = self._generate_input()
        weights = WeightGenerator.generate(self.model_spec)

        # Apply quantization if needed
        if self.quantization == 'golden-stick':
            self.quantizer.quant(input_data, weights, self.quant_model_dir)
            weights = self._load_quant_weights()

        # Create network and load weights
        network = self._create_network()
        weights = self._prepare_weights(weights)
        WeightLoader.load_into_network(network, weights)

        # Process weights after loading (e.g., format conversion for custom ops)
        if hasattr(network, 'process_weights_after_loading'):
            network.process_weights_after_loading()

        # Run inference
        net_input = self._create_tensor(input_data)
        output_dict = network.forward(net_input)

        # Save output
        if self.rank_id is None or int(self.rank_id) == 0:
            np.savez(self.config.output_path, **output_dict)

    def _generate_input(self) -> np.ndarray:
        """Generate random input data."""
        np.random.seed(42)
        batch_size = 4
        return 0.01 * np.random.randn(batch_size, self.model_spec.linear_specs[0].input_size).astype(np.float32)

    def _create_tensor(self, data: np.ndarray) -> ms.Tensor:
        """Create a MindSpore tensor with appropriate dtype."""
        dtype = self.model_spec.compute_dtype
        if isinstance(dtype, str):
            dtype = convert_dtype_str_to_ms(dtype)
        return ms.Tensor(data, dtype=dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simple MCore parallel linear test with YAML configuration")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("--tensor_parallel", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--output_path", type=str, default="output.npz",
                        help="Output file path")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization method (e.g., 'golden-stick')")

    args = parser.parse_args()

    ms.set_context(device_target="Ascend",
                   mode=ms.GRAPH_MODE,
                   jit_config={"jit_level": "O0", "infer_boost": "on"},
                   deterministic="ON")

    runner = SimpleMCoreRunner(args)
    runner.run()
