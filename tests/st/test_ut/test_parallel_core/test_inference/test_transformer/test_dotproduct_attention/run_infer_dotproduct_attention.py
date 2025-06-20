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
"""Run DotProduct Attention UT of inference with configurable parameters via args"""
import argparse
import os
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import mint
from mindspore.communication import init

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.parallel_core.inference.transformer.dot_product_attention import  DotProductAttention
from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_dotproduct_attention.data_gen_utils import get_init_params


class DotProductAttentionRunner:
    """Class to manage Dopt_Attn"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.batch_size = self.args.batch_size
        self.seq_length = self.args.seq_length
        self.num_heads = self.args.num_heads
        self.hidden_size = self.args.hidden_size

        self.config = self.get_dopt_attention_config()

        init_params = get_init_params(num_kv_heads=self.args.num_query_groups)

        self.dopt_attn_mask = mint.ones((self.seq_length, self.seq_length), dtype=mstype.float16)
        self.dopt_attn_mask = mint.triu(self.dopt_attn_mask, diagonal=1).astype(mstype.bfloat16)
        self.query = ms.Tensor(init_params.get("query"), dtype=mstype.bfloat16)
        self.key = ms.Tensor(init_params.get("key"), dtype=mstype.bfloat16)
        self.value = ms.Tensor(init_params.get("value"), dtype=mstype.bfloat16)

        # RANK_ID and worker_num are set by msrun environment
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        # Set parallel context
        if self.rank_id is not None:
            init()
            initialize_model_parallel(tensor_model_parallel_size=1)

    def get_dopt_attention_config(self):
        """Generate config for DotProductAttention test."""
        config = TransformerConfig(
            num_attention_heads=self.num_heads,
            num_query_groups=self.args.num_query_groups,
            hidden_size=self.hidden_size,
            sequence_parallel=False,
            num_layers=1,
            compute_dtype='bf16',
            softmax_compute_dtype='bf16'
        )

        return config

    def build_model(self):
        """Build Dopt_Attn"""
        net = DotProductAttention(
            config=self.config,
            layer_number=1,
            attn_mask_type=None
        )
        return net

    def run(self):
        """Run the dopt_attn with given inputs"""
        net = self.build_model()

        output = net(
            query_layer=self.query,
            key_layer=self.key,
            value_layer=self.value,
            attention_mask=None
        )
        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.astype(mstype.float16).asnumpy() for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run DoptAttention test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_query_groups", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default="output_ms_dpa.npz")

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

    runner = DotProductAttentionRunner(args)
    # Execute Runner
    runner.run()


if __name__ == "__main__":
    main()
