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
"""Run CrossEntropyLoss accuracy test with configurable parameters via args"""
import os
import argparse
from pathlib import Path
import numpy as np

from data_gen_utils import get_init_params, get_cpu_output, get_static_output

import mindspore as ms

from mindformers.pynative.loss import CrossEntropyLoss


SCRIPT_DIR = Path(__file__).parent.resolve()


class CrossEntropyLossRunner:
    """Class to manage CrossEntropyLoss model and data"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.calculate_per_token_loss = self.args.calculate_per_token_loss

        self.vocab_size = self.args.vocab_size
        self.batch_size = self.args.batch_size
        self.seq_length = self.args.seq_length

        logits, target, input_mask = get_init_params(self.batch_size, self.seq_length, self.vocab_size).values()
        self.output_cpu = get_cpu_output(logits, target, input_mask)

        self.logits = ms.Tensor(logits, dtype=ms.float32)
        self.target = ms.Tensor(
            target.reshape((self.batch_size, self.seq_length)).reshape(-1),
            dtype=ms.int32,
        )
        self.input_mask = ms.Tensor(
            input_mask.reshape((self.batch_size, self.seq_length)).reshape(-1),
            dtype=ms.int32,
        )

    def run(self):
        """Run the model with given inputs"""
        ms.set_context(mode=1)
        net = CrossEntropyLoss(
            calculate_per_token_loss=self.calculate_per_token_loss,
        )

        grad_fn = ms.value_and_grad(net, grad_position=0)
        result, grad = grad_fn(self.logits, self.target, self.input_mask)

        output_pynative = {}
        if not self.calculate_per_token_loss:
            output_pynative["loss"] = result
        else:
            numerator, denominator = result
            output_pynative["numerator"] = numerator
            output_pynative["denominator"] = denominator
        output_pynative["grad"] = grad
        output_pynative = {k: v.asnumpy().astype(np.float32) for k, v in output_pynative.items() if v is not None}
        output_static = get_static_output(self.logits, self.target, self.input_mask)
        output_path = self.args.output_path
        np.savez(os.path.join(output_path, "output_pynative_loss.npz"), **output_pynative)
        np.savez(os.path.join(output_path, "output_static_loss.npz"), **output_static)
        np.savez(os.path.join(output_path, "output_cpu_loss.npz"), **self.output_cpu)

def main():
    parser = argparse.ArgumentParser(description="Run CrossEntropyLoss test")
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=8)
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--calculate_per_token_loss", type=lambda x: x.lower() == "true", default="false")

    args = parser.parse_args()

    ms.context.set_context(deterministic="ON")
    ms.set_seed(42)

    runner = CrossEntropyLossRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
