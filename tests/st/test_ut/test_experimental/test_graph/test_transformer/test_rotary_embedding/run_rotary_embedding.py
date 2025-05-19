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
"""Run RotaryEmbedding accuracy test with configurable parameters via args"""
import argparse
import os
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore.communication import init
from mindformers.experimental.graph.transformer.rotary_pos_embedding import RotaryEmbedding
from data_gen_utils import get_init_params
SCRIPT_DIR = Path(__file__).parent.resolve()

class RotaryEmbeddingRunner:
    """Class to manage RotaryEmbedding model and weights"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.kv_channels = self.args.kv_channels
        self.rotary_percent = self.args.rotary_percent
        self.rotary_interleaved = self.args.rotary_interleaved
        self.seq_len_interpolation_factor = self.args.seq_len_interpolation_factor
        self.rope_scaling = self.args.rope_scaling

        self.compute_dtype = ms.bfloat16
        self.param_init_dtype = ms.float32

        init_params = get_init_params()

        self.max_seq_len = init_params.get("max_seq_len")
        self.offset = init_params.get("offset")

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

    def build_model(self):
        """Build and initialize RotaryEmbedding model"""
        net = RotaryEmbedding(
            kv_channels=self.kv_channels,
            rotary_percent=self.rotary_percent,
            rotary_interleaved=self.rotary_interleaved,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            rope_scaling=self.rope_scaling,
        )
        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()

        output, _ = net(self.max_seq_len, self.offset)
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
    parser = argparse.ArgumentParser(description="Run RotaryEmbedding test")
    parser.add_argument("--kv_channels", type=int, default=32)
    parser.add_argument("--rotary_percent", type=float, default=1.0)
    parser.add_argument("--seq_len_interpolation_factor", type=float, default=None)
    parser.add_argument("--rotary_interleaved", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--rope_scaling", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--output_path", type=str, default="output_ms.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)

    args = parser.parse_args()

    ms.set_deterministic(True)
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_seed(42)

    # Prepare input
    runner = RotaryEmbeddingRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
