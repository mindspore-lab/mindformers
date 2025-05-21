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
"""Run FusedScaleMaskSoftmax accuracy test with configurable parameters via args"""
import argparse
import os
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore.communication import init
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.fused_softmax import FusedScaleMaskSoftmax
from mindformers.parallel_core.training_graph.transformer.utils import get_attn_mask_func
from mindformers.parallel_core.training_graph.transformer.enums import AttnMaskType
from mindformers.parallel_core.utils.init_method import init_method_normal

from data_gen_utils import get_init_params

SCRIPT_DIR = Path(__file__).parent.resolve()

class FusedSoftmaxRunner:
    """Class to manage FusedScaleMaskSoftmax model execution."""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.batch_size = self.args.batch_size
        self.num_heads = self.args.num_heads
        self.seq_length = self.args.seq_length
        self.head_dim = self.args.seq_length

        if self.args.input_in_bf16:
            self.compute_dtype = ms.bfloat16
        elif self.args.input_in_fp16:
            self.compute_dtype = ms.float16
        else:
            self.compute_dtype = ms.float32 # Default if neither is specified

        init_data = get_init_params(
            self.batch_size, self.num_heads, self.seq_length)

        if self.compute_dtype == ms.bfloat16:
            self.inputs = ms.Tensor(init_data["inputs"], dtype=ms.bfloat16)
        elif self.compute_dtype == ms.float16:
            self.inputs = ms.Tensor(init_data["inputs"].astype(np.float16), dtype=ms.float16)
        else:
            self.inputs = ms.Tensor(init_data["inputs"], dtype=ms.float32)


        self.construct_mask_tensor = None
        if self.args.use_construct_mask:
            self.construct_mask_tensor = ms.Tensor(init_data["external_mask"], dtype=ms.int32)

        rank_id_str: str | None = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None
        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        if self.rank_id is not None:
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True,
                # grad_accumulation_step=1 # If needed
            )
            init()

        self.config = TransformerConfig(
            data_parallel_size=self.worker_num // self.args.tensor_parallel,
            tensor_model_parallel_size=self.args.tensor_parallel,
            compute_dtype=self.compute_dtype,
            num_attention_heads=self.num_heads,
            init_method=init_method_normal(0.01, ms.float32),
            output_layer_init_method=init_method_normal(0.01, ms.float32),
        )

    def build_model(self):
        """Build and initialize FusedScaleMaskSoftmax model."""
        selected_mask_func = get_attn_mask_func(self.args.mask_func_name)

        # Convert string scale to float or None
        scale_val = None
        if self.args.scale.lower() != "none":
            try:
                scale_val = float(self.args.scale)
            except ValueError:
                raise ValueError(f"Invalid scale value: {self.args.scale}. Must be a float or 'None'.")

        net = FusedScaleMaskSoftmax(
            input_in_fp16=self.args.input_in_fp16,
            input_in_bf16=self.args.input_in_bf16,
            attn_mask_type=self.args.attn_mask_type,
            mask_func=selected_mask_func(self.config),
            softmax_in_fp32=self.args.softmax_in_fp32,
            scale=scale_val,
            config=self.config,
        )
        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()
        # Pass the generated mask if use_construct_mask is True
        mask_to_pass = self.construct_mask_tensor if self.args.use_construct_mask else None
        output = net(self.inputs, mask_to_pass)
        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            # Convert to float32 for saving, common practice for bf16/fp16
            output_np = {k: v.asnumpy().astype(np.float32) for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)

def main():
    parser = argparse.ArgumentParser(description="Run FusedScaleMaskSoftmax test")
    # Input shape parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=4)
    # Model configuration parameters from table
    parser.add_argument("--input_in_fp16", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--input_in_bf16", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument(
        "--attn_mask_type",
        type=lambda s: AttnMaskType[s],
        default=AttnMaskType.causal,
        choices=list(AttnMaskType),
        help="Type of attention mask to use"
    )
    parser.add_argument("--mask_func_name", type=str, default="attn_mask_fill", choices=["attn_mask_fill"])
    parser.add_argument("--softmax_in_fp32", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--scale", type=str, default="None") # Keep as string to handle "None"
    parser.add_argument("--use_construct_mask", type=lambda x: x.lower() == "true", default=False) # For construct mask input
    # Output and parallelism
    parser.add_argument("--output_path", type=str, default="output_softmax_ms.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    args = parser.parse_args()

    ms.context.set_context(deterministic="ON")
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_seed(42)

    runner = FusedSoftmaxRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
