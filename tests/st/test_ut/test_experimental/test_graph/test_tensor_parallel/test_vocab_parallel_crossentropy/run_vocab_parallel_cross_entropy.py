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
"""Run VocabParallelCrossEntropy accuracy test with configurable parameters via args"""

import argparse
import os
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore.communication import init
from mindformers.experimental.graph.loss_func import VocabParallelCrossEntropy
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from data_gen_utils import get_init_params

SCRIPT_DIR = Path(__file__).parent.resolve()


class VocabParallelCrossEntropyRunner:
    """Class to manage VocabParallelCrossEntropy model and data"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.check_for_nan_in_loss_and_grad = self.args.check_for_nan_in_loss_and_grad
        self.calculate_per_token_loss = self.args.calculate_per_token_loss

        self.vocab_size = self.args.vocab_size
        self.batch_size = self.args.batch_size
        self.seq_length = self.args.seq_length

        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        if self.rank_id is not None:
            ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
            init()

        self.config = TransformerConfig(
            data_parallel=self.worker_num // self.args.tensor_parallel,
            tensor_parallel=self.args.tensor_parallel,
            num_attention_heads=self.args.tensor_parallel,
        )

        init_params_data = get_init_params(self.batch_size, self.seq_length, self.vocab_size)

        logits = init_params_data.get("logits")

        self.logits = ms.Tensor(logits, dtype=ms.float32)
        self.target = ms.Tensor(
            init_params_data.get("target").reshape((self.batch_size, self.seq_length)).transpose((1, 0)).reshape(-1),
            dtype=ms.int32,
        )
        self.input_mask = ms.Tensor(
            init_params_data.get("input_mask")
            .reshape((self.batch_size, self.seq_length))
            .transpose((1, 0))
            .reshape(-1),
            dtype=ms.int32,
        )

    def build_model(self):
        """Build VocabParallelCrossEntropy model"""
        net = VocabParallelCrossEntropy(
            parallel_config=self.config,
            check_for_nan_in_loss_and_grad=self.check_for_nan_in_loss_and_grad,
            calculate_per_token_loss=self.calculate_per_token_loss,
        )
        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()

        result = net(self.logits, self.target, self.input_mask)

        output_ms = {}
        if not self.calculate_per_token_loss:
            output_ms["loss"] = result
        else:
            numerator, denominator = result
            output_ms["numerator"] = numerator
            output_ms["denominator"] = denominator

        if self.rank_id is None or self.rank_id == 0:
            output_np = {k: v.asnumpy().astype(np.float32) for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run VocabParallelCrossEntropy test")
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=8)
    parser.add_argument("--check_for_nan_in_loss_and_grad", type=lambda x: x.lower() == "true", default="false")
    parser.add_argument("--calculate_per_token_loss", type=lambda x: x.lower() == "true", default="false")
    parser.add_argument("--output_path", type=str, default="output_ms_loss.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)

    args = parser.parse_args()

    ms.context.set_context(deterministic="ON")
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_seed(42)

    runner = VocabParallelCrossEntropyRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
