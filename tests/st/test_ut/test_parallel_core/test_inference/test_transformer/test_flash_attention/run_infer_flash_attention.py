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
"""Run Flash Attention UT of inference with configurable parameters via args"""
import argparse
import os
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Tensor, mint
from mindspore.communication import init

from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.parallel_core.inference.transformer.flash_attention import FlashAttention
from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_flash_attention.data_gen_utils import (
    get_init_params,
    NUM_BLOCKS,
    BLOCK_SIZE,
)


class FlashAttentionRunner:
    """Class to manage Flash_Attn"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.num_heads = self.args.num_heads
        self.batch_size = self.args.batch_size
        self.is_prefill = self.args.is_prefill
        self.seq_length = self.args.seq_length
        self.hidden_size = self.args.hidden_size
        self.n_kv_heads = self.args.n_kv_heads
        self.head_dim = int(self.hidden_size / self.num_heads)
        self.keep_prob = self.args.keep_prob
        self.scale_value = self.args.scale_value

        self.kv_cache_shape = (
            NUM_BLOCKS, BLOCK_SIZE, self.n_kv_heads, self.head_dim
        )

        init_params = get_init_params(is_prefill=self.is_prefill,
                                      n_kv_heads=self.n_kv_heads)

        self.block_tables = Tensor(np.ones((self.batch_size, BLOCK_SIZE)) * -1, dtype=mstype.int32)
        self.block_tables[0][0] = 0

        if self.is_prefill:
            self.slot_mapping = Tensor(np.arange(self.batch_size * self.seq_length), dtype=mstype.int32)
            self.batch_valid_length = Tensor(np.ones((self.seq_length,)), dtype=mstype.int32)
        else:
            self.slot_mapping = Tensor(np.arange(self.batch_size * 1), dtype=mstype.int32)
            self.batch_valid_length = Tensor(np.ones((1,)), dtype=mstype.int32)

        self.flash_attn_mask = mint.ones((self.seq_length, self.seq_length), dtype=mstype.float16)
        self.flash_attn_mask = mint.triu(self.flash_attn_mask, diagonal=1).astype(mstype.bfloat16)

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

    def build_model(self):
        """Build Flash_Attn"""
        net = FlashAttention(
            head_num=self.num_heads,
            head_dim=self.head_dim,
            keep_prob=self.keep_prob,
            kv_head_num=self.n_kv_heads,
            scale_value=self.scale_value
        )
        return net

    def run(self):
        """Run the flash_attn with given inputs"""
        net = self.build_model()
        if not self.is_prefill:
            net.phase = "increment"
            net.add_flags(is_prefill=False)

        key_cache = mint.zeros(self.kv_cache_shape, dtype=mstype.bfloat16)
        value_cache = mint.zeros(self.kv_cache_shape, dtype=mstype.bfloat16)

        output = net(
            query=self.query,
            key=self.key,
            value=self.value,
            kv_cache=None,
            slot_mapping=self.slot_mapping,
            block_tables=self.block_tables,
            actual_seq_qlen=self.batch_valid_length,
            actual_seq_kvlen=self.batch_valid_length,
            batch_valid_length=self.batch_valid_length,
            context_lens_tensor=self.batch_valid_length,
            key_cache=key_cache,
            value_cache=value_cache
        )
        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.asnumpy().astype(np.float16) for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run FA test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--n_kv_heads", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--keep_prob", type=float, default=1.0)
    parser.add_argument("--scale_value", type=float, default=0.25)
    parser.add_argument("--is_prefill", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--output_path", type=str, default="output_ms_fa.npz")

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

    runner = FlashAttentionRunner(args)
    # Execute Runner
    runner.run()


if __name__ == "__main__":
    main()
