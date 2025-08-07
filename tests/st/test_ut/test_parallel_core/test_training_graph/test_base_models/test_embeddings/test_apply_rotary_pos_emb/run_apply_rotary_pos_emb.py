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
"""Run ApplyRotaryPosEmb accuracy test with configurable parameters via args"""
import argparse
import os
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore.communication import init
from mindformers.parallel_core.training_graph.base_models.common.embeddings.rope_utils import ApplyRotaryPosEmb
from mindformers.parallel_core.transformer_config import TransformerConfig
from data_gen_utils import get_init_params
SCRIPT_DIR = Path(__file__).parent.resolve()


class ApplyRotaryPosEmbRunner:
    """Class to manage ApplyRotaryPosEmb model and weights"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.rotary_interleaved = self.args.rotary_interleaved
        self.multi_latent_attention = self.args.multi_latent_attention

        self.compute_dtype = ms.bfloat16
        self.param_init_dtype = ms.float32

        init_params = get_init_params()

        self.input_t = ms.Tensor(init_params.get("t"), dtype=ms.bfloat16)
        self.input_freqs = ms.Tensor(init_params.get("freqs"), dtype=ms.bfloat16)
        self.mscale = 1.0
        self.freqs = (self.input_freqs, self.mscale)

        # RANK_ID and worker_num are set by msrun environment
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))


        # Set parallel context
        if self.rank_id is not None:
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True
            )
            init()

        # Transformer config
        self.config = TransformerConfig(
            data_parallel_size=self.worker_num // self.args.tensor_parallel,
            tensor_model_parallel_size=self.args.tensor_parallel,
            compute_dtype='bf16',
            params_dtype='fp32',
            num_attention_heads=self.args.tensor_parallel,
            num_layers=1
        )

    def build_model(self):
        """Build and initialize ApplyRotaryPosEmb model"""
        net = ApplyRotaryPosEmb(
            config=self.config,
        )
        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()

        output = net(self.input_t, self.freqs, self.rotary_interleaved, self.multi_latent_attention)
        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {
                k: v.asnumpy().astype(np.float32)
                for k, v in output_ms.items()
                if v is not None
            }
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run ApplyRotaryPosEmb test")
    parser.add_argument("--rotary_interleaved", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--multi_latent_attention", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--output_path", type=str, default="output_ms.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)

    args = parser.parse_args()

    ms.set_deterministic(True)
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_seed(42)

    # Prepare input
    runner = ApplyRotaryPosEmbRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
