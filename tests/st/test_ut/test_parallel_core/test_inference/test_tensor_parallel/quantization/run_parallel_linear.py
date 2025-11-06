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
"""Run ColumnParallelLinear accuracy test with configurable parameters via args"""


import argparse
import glob
import os
import tempfile

import numpy as np
from safetensors import safe_open

import mindspore as ms
from mindspore.communication import init
from numpy_quantizer import NumpyQuantizer
from gpt_model_for_test import GPTModelForTest, LinearSpec, ModelSpec
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.parallel_core.process_group_config import ModelCommProcessGroups


class ParallelModelRunner:
    """Runner for parallel model testing with quantization support."""

    def __init__(self, config):
        """Initialize the parallel model runner with given arguments."""
        self.config = config
        # set up parallel context
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None
        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))
        self.model_comm_pgs = ModelCommProcessGroups.get_default_model_comm_pgs()
        if self.rank_id is not None:
            init()
            initialize_model_parallel(tensor_model_parallel_size=self.config.tensor_parallel)
            self.model_comm_pgs = ModelCommProcessGroups.use_parallel_state_groups(required_groups=['tp'])

        linear_specs = []
        quant_policys = []
        self.quantization = config.quantization
        for linear_type in config.linear_types:
            for has_bias in [True, False]:
                for quant_policy in config.quant_policies:
                    quant_policy = quant_policy if config.quantization == 'golden-stick' else 'float'
                    linear_specs.append(LinearSpec(linear_type, config.input_size, config.output_size,
                                                has_bias, config.compute_dtype, quant_policy))
                    quant_policys.append(quant_policy)

        self.model_spec = ModelSpec(
            compute_dtype=config.compute_dtype,
            param_init_dtype=config.param_init_dtype,
            tensor_parallel=config.tensor_parallel,
            linear_specs=linear_specs,
        )
        self.quant_model_dir = None
        if self.quantization == 'golden-stick':
            self.quantizer = NumpyQuantizer(self.model_spec, quant_policys)
            self.quant_model_dir = tempfile.mkdtemp(prefix="quant_model_for_test_")

    @staticmethod
    def _gen_float_weights(model_spec):
        """Generate random float weights for model specifications."""
        np.random.seed(42)
        weights = {}
        for index, linear_spec in enumerate(model_spec.linear_specs):
            weight_shape = (linear_spec.output_size, linear_spec.input_size)
            output_size = linear_spec.output_size
            weight = 0.01 * np.random.randn(*weight_shape).astype(np.float32)
            weights[f"linears.{index}.weight"] = weight
            if linear_spec.has_bias:
                bias = 0.01 * np.random.randn(output_size).astype(np.float32)
                weights[f"linears.{index}.bias"] = bias
        return weights

    @staticmethod
    def _gen_input(model_spec):
        """Generate random input data for model specifications."""
        np.random.seed(42)
        return 0.01 * np.random.randn(2 * 2, model_spec.linear_specs[0].input_size).astype(np.float32)

    def _create_network(self):
        """Create the network model for testing."""
        return GPTModelForTest(self.model_spec, self.model_comm_pgs, self.quantization, self.quant_model_dir)

    def _load_quant_weights(self):
        """Load quantized weights from the model directory."""
        if not os.path.isdir(self.quant_model_dir):
            raise ValueError(f"Invalid quant_model_dir: {self.quant_model_dir}")
        safetensor_files = glob.glob(os.path.join(self.quant_model_dir, "*.safetensors"))
        if len(safetensor_files) == 1:
            safetensor_file = safetensor_files[0]
        elif len(safetensor_files) > 1:
            raise FileNotFoundError(f"Found multiple safetensor files in {self.quant_model_dir}")
        else:
            raise FileNotFoundError(f"Found no safetensor file in {self.quant_model_dir}")
        if not os.path.exists(safetensor_file):
            raise FileNotFoundError(f"File {safetensor_file} not found.")
        with safe_open(safetensor_file, framework="np", device="cpu") as f:
            weights = {}
            for key in f.keys():
                weights[key] = f.get_slice(key)
        return weights

    @staticmethod
    def load_weights_into_network(network, weights):
        """Load weights into the network parameters."""
        params = network.parameters_dict()
        loaded = []
        for k, v in weights.items():
            param = params.get(k)
            if param is None:
                continue
            loaded.append(k)
            param.weight_loader(param, v)
        print(f"weights not use: {set(weights.keys()) - set(loaded)}", flush=True)
        print(f"params not load: {set(params.keys()) - set(loaded)}", flush=True)

    def run(self):
        """Run the parallel model test."""
        input_data = ParallelModelRunner._gen_input(self.model_spec)
        weights = ParallelModelRunner._gen_float_weights(self.model_spec)
        if self.quantization == 'golden-stick':
            self.quantizer.quant(input_data, weights, self.quant_model_dir)
            weights = self._load_quant_weights()
        network = self._create_network()
        ParallelModelRunner.load_weights_into_network(network, weights)
        net_input = ms.Tensor(input_data, dtype=LinearSpec.convert_pt_dtype_to_ms(self.model_spec.compute_dtype))
        output_dict = network.forward(net_input)

        if self.rank_id is None or int(self.rank_id) == 0:
            np.savez(self.config.output_path, **output_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ColumnParallelLinear test")
    parser.add_argument("--linear_types", type=str, action='append', default=None,
                        help="List of linear types, e.g., --linear_types ColumnParallelLinear "\
                             "--linear_types RowParallelLinear")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--compute_dtype", type=str, default='bf16')
    parser.add_argument("--param_init_dtype", type=str, default='bf16')
    parser.add_argument("--output_path", type=str, default="output.npz")
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--quant_policies", type=str, action='append', default=None,
                        help="List of quantization policies, e.g., --quant_policies a8w8 --quant_policies a8dynw8")
    args = parser.parse_args()
    args.input_size = 32
    args.output_size = 32

    ms.set_context(device_target="Ascend",
                   mode=ms.GRAPH_MODE,
                   jit_config={"jit_level": "O0", "infer_boost": "on"},
                   deterministic="ON")

    quant_runner = ParallelModelRunner(args)
    quant_runner.run()
