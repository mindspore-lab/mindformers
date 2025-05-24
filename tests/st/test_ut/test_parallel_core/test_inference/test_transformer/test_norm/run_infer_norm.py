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
"""Run mcore norm UT of inference with configurable parameters via args"""
import argparse
import os
from pathlib import Path
import numpy as np

import mindspore as ms
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype
from mindspore.communication import init

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.parallel_core.inference.transformer.norm import get_norm_cls
from mindformers.parallel_core.utils.spec_utils import build_module

from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_norm.data_gen_utils import (
    get_init_params,
)

SCRIPT_DIR = Path(__file__).parent.resolve()


class LayerNormRunner:
    """Class to manage LayerNorm module"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.module = self.args.module
        self.batch_size = self.args.batch_size
        self.seq_length = self.args.seq_length
        self.hidden_size = self.args.hidden_size
        self.eps = self.args.eps

        init_params = get_init_params(self.batch_size, self.seq_length, self.hidden_size)

        self.input = Tensor(init_params.get("input"), dtype=mstype.bfloat16)
        self.weight = init_params.get("weight")

        # RANK_ID and worker_num are set by msrun environment
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        # Set parallel context
        if self.rank_id is not None:
            init()
            initialize_model_parallel(tensor_model_parallel_size=1)

        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=self.hidden_size,
            num_attention_heads=2,
            layernorm_compute_dtype="bfloat16",
            normalization="LayerNorm"
        )

    def build_model(self):
        """Build LayerNorm module"""
        net = build_module(
            get_norm_cls(self.config),
            self.config,
            hidden_size=self.hidden_size,
            eps=self.eps
        )
        param_dict = {
            "gamma": Parameter(self.weight)
        }
        self._load_weights(net, param_dict)
        return net

    def _load_weights(self, net, param_dict):
        """load weights for LayerNorm module"""
        ms.load_param_into_net(net, param_dict)

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()

        output = net(self.input)
        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.astype(mstype.float16).asnumpy() for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run norm test")
    parser.add_argument("--module", type=str, default="LayerNorm")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--output_path", type=str, default="output_ms_layernorm.npz")

    args = parser.parse_args()

    ms.context.set_context(deterministic="ON")
    jit_level = "O0"
    infer_boost = "on"
    ms.set_context(device_target="Ascend",
                   mode=ms.GRAPH_MODE,
                   jit_config={"jit_level": jit_level, "infer_boost": infer_boost})
    seed_value = 2025
    ms.set_seed(seed_value)
    np.random.seed(seed_value)

    # Init Runner to prepare inputs
    if args.module == "LayerNorm":
        runner = LayerNormRunner(args)
    else:
        raise ValueError(f"The activation unit test currently supports only LayerNorm, "
                         f"but got {args.module}")
    # Execute Runner
    runner.run()


if __name__ == "__main__":
    main()
