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
"""Run CausalMaskGenerate accuracy test with configurable parameters via args"""
import argparse
import os
from pathlib import Path
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.communication import init

from data_gen_utils import get_init_params
from mindformers.pynative.layers.mask_generate import CausalMaskGenerate

SCRIPT_DIR = Path(__file__).parent.resolve()

class CausalMaskGenerateRunner:
    """Class to manage CausalMaskGenerate model and inputs"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        # Parameters that need testing
        self.use_attn_mask_compression = self.args.use_attn_mask_compression
        self.seq_length = self.args.seq_length
        self.is_dynamic = self.args.is_dynamic
        use_tokens = self.args.use_tokens

        # Parameters with default values (not tested)
        batch_size = self.args.batch_size
        compute_type = self.args.compute_type
        self.compute_type = getattr(mstype, compute_type)
        self.pad_token_id = 0

        # If use_tokens is True, masks will be None
        # If use_tokens is False, masks must be provided (handled in get_init_params)
        init_params = get_init_params(batch_size, self.seq_length, use_tokens=use_tokens)

        # tokens dtype: int32 (token IDs are integers)
        # masks dtype: same as compute_type (will be cast to compute_type in the code)
        tokens_data = init_params.get("tokens")
        self.tokens = ms.Tensor(tokens_data, dtype=ms.int32) if tokens_data is not None else None
        masks_data = init_params.get("masks")
        self.masks = ms.Tensor(masks_data, dtype=self.compute_type) if masks_data is not None else None

        # RANK_ID and worker_num are set by msrun environment
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        # Set parallel context for multi-card
        if self.rank_id is not None:
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True
            )
            init()

    def build_model(self):
        """Build and initialize CausalMaskGenerate model"""
        net = CausalMaskGenerate(
            seq_length=self.seq_length,
            compute_type=self.compute_type,
            is_dynamic=self.is_dynamic,
            pad_token_id=self.pad_token_id,
            use_attn_mask_compression=self.use_attn_mask_compression,
        )
        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()

        output = net(self.tokens, self.masks)
        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.asnumpy().astype(np.float32) for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run CausalMaskGenerate test")
    # Parameters that need testing
    parser.add_argument("--seq_length", type=int, default=8)
    parser.add_argument("--is_dynamic", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--use_attn_mask_compression", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--use_tokens", type=lambda x: x.lower() == "true", default=True)
    # Parameters with default values (not tested, kept for compatibility)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--compute_type", type=str, default="float16")
    parser.add_argument("--output_path", type=str, default="output_ms.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)

    args = parser.parse_args()

    ms.set_deterministic(True)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms.set_seed(42)

    # Prepare input
    runner = CausalMaskGenerateRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
