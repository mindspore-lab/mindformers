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
from mindspore import Tensor, mint, ops
from mindspore.communication import init

from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.parallel_core.inference.transformer.flash_attention import FlashAttention
from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_flash_attention.data_gen_utils import (
    get_init_params,
    NUM_BLOCKS,
    BLOCK_SIZE,
    BATCH_SIZE,
    SEQ_LENGTH,
    NUM_HEADS,
    HIDDEN_SIZE
)


class FlashAttentionRunner:
    """Class to manage Flash_Attn"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.num_heads = self.args.num_heads
        self.batch_size = self.args.batch_size
        self.is_prefill = True
        self.prefill_seq_length = self.args.seq_length[1]
        self.prefill_seq_length = int(self.prefill_seq_length)
        self.decoder_seq_length = self.args.seq_length[-2]
        self.decoder_seq_length = int(self.decoder_seq_length)
        self.hidden_size = self.args.hidden_size
        self.n_kv_heads = self.args.n_kv_heads
        self.head_dim = int(self.hidden_size / self.num_heads)
        self.keep_prob = self.args.keep_prob
        self.scale_value = self.args.scale_value

        self.kv_cache_shape = (
            NUM_BLOCKS, BLOCK_SIZE, self.n_kv_heads, self.head_dim
        )
        self.block_tables = Tensor(np.ones((self.batch_size, 11)) * -1, dtype=mstype.int32)
        self.block_tables[0][0] = 2
        self.block_tables[1][0] = 1
        init_params = get_init_params(n_kv_heads=self.n_kv_heads)

        assert self.num_heads % self.n_kv_heads == 0, (
            "n_kv_heads must be divided by num_heads!"
        )

        self.prefill_query = ms.Tensor(init_params.get("prefill_query"),
                                       dtype=mstype.bfloat16).reshape(1, BATCH_SIZE * SEQ_LENGTH[0],
                                                                      NUM_HEADS * int(HIDDEN_SIZE / NUM_HEADS))
        self.prefill_key = ms.Tensor(init_params.get("prefill_key"),
                                     dtype=mstype.bfloat16).reshape(1, BATCH_SIZE * SEQ_LENGTH[0],
                                                                    self.n_kv_heads * int(HIDDEN_SIZE / NUM_HEADS))
        self.prefill_value = ms.Tensor(init_params.get("prefill_value"),
                                       dtype=mstype.bfloat16).reshape(1, BATCH_SIZE * SEQ_LENGTH[0],
                                                                      self.n_kv_heads * int(
                                                                          HIDDEN_SIZE / NUM_HEADS))
        self.decoder_query = ms.Tensor(init_params.get("decoder_query"),
                                       dtype=mstype.bfloat16).reshape(1, BATCH_SIZE * min(1, SEQ_LENGTH[1]),
                                                                      NUM_HEADS * int(HIDDEN_SIZE / NUM_HEADS))
        self.decoder_key = ms.Tensor(init_params.get("decoder_key"),
                                     dtype=mstype.bfloat16).reshape(1, BATCH_SIZE * min(1, SEQ_LENGTH[1]),
                                                                    self.n_kv_heads * int(HIDDEN_SIZE / NUM_HEADS))
        self.decoder_value = ms.Tensor(init_params.get("decoder_value"),
                                       dtype=mstype.bfloat16).reshape(1, BATCH_SIZE * min(1, SEQ_LENGTH[1]),
                                                                      self.n_kv_heads * int(
                                                                          HIDDEN_SIZE / NUM_HEADS))

        self.prefill_slot_mapping = Tensor(init_params.get("prefill_slot_mapping"), dtype=mstype.int32)
        self.decoder_slot_mapping = Tensor(init_params.get("decoder_slot_mapping"), dtype=mstype.int32)

        self.reshape_and_cache = ops.auto_generate.ReshapeAndCache()
        self.key_cache = mint.zeros(self.kv_cache_shape, dtype=mstype.bfloat16)
        self.value_cache = mint.zeros(self.kv_cache_shape, dtype=mstype.bfloat16)

        self.get_slot_map()
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        # Set parallel context
        if self.rank_id is not None:
            init()
            initialize_model_parallel(tensor_model_parallel_size=1)

    def get_slot_map(self):
        if self.is_prefill:
            self.flash_attn_mask = mint.ones((self.prefill_seq_length, self.prefill_seq_length), dtype=mstype.float16)
            self.flash_attn_mask = mint.triu(self.flash_attn_mask, diagonal=1).astype(mstype.bfloat16)
            self.batch_valid_length = Tensor(np.ones((self.batch_size,)) * SEQ_LENGTH[0], dtype=mstype.int32)
        else:
            self.flash_attn_mask = mint.ones((self.decoder_seq_length, self.decoder_seq_length), dtype=mstype.float16)
            self.flash_attn_mask = mint.triu(self.flash_attn_mask, diagonal=1).astype(mstype.bfloat16)
            self.batch_valid_length = Tensor(np.ones((self.batch_size,)) * (SEQ_LENGTH[0] + SEQ_LENGTH[1]),
                                             dtype=mstype.int32)

    def build_model(self):
        """Build Flash_Attn"""
        net = FlashAttention(
            head_num=self.num_heads,
            head_dim=self.head_dim,
            keep_prob=self.keep_prob,
            kv_head_num=self.n_kv_heads,
            pa_kv_head_num=self.n_kv_heads,
            scale_value=self.scale_value
        )
        return net

    def run(self):
        """Run the flash_attn with given inputs"""
        # prefill
        net = self.build_model()
        if not self.is_prefill:
            net.phase = "increment"
            net.add_flags(is_prefill=False)

        prefill_output = net(
            query=self.prefill_query,
            key=self.prefill_key,
            value=self.prefill_value,
            slot_mapping=self.prefill_slot_mapping,
            block_tables=self.block_tables,
            actual_seq_qlen=self.batch_valid_length,
            actual_seq_kvlen=self.batch_valid_length,
            batch_valid_length=self.batch_valid_length,
            context_lens_tensor=self.batch_valid_length,
            attn_mask=self.flash_attn_mask,
            key_cache=self.key_cache,
            value_cache=self.value_cache
        )

        # decode
        self.is_prefill = False
        self.get_slot_map()
        if not self.is_prefill:
            net.phase = "increment"
            net.add_flags(is_prefill=False)

        self.reshape_and_cache(self.prefill_key, self.prefill_value, self.key_cache, self.value_cache,
                               self.prefill_slot_mapping)

        decode_output = net(
            query=self.decoder_query,
            key=self.decoder_key,
            value=self.decoder_value,
            slot_mapping=self.decoder_slot_mapping,
            block_tables=self.block_tables,
            actual_seq_qlen=self.batch_valid_length,
            actual_seq_kvlen=self.batch_valid_length,
            batch_valid_length=self.batch_valid_length,
            context_lens_tensor=self.batch_valid_length,
            key_cache=self.key_cache,
            value_cache=self.value_cache
        )

        output_ms = {"prefill_output": prefill_output, "decode_output": decode_output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.asnumpy().astype(np.float16) for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run FA test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=list[int], default=[2, 1])
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=64)  # 32
    parser.add_argument("--keep_prob", type=float, default=1.0)
    parser.add_argument("--scale_value", type=float, default=0.25)
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
