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
"""Test Attention"""
import argparse
import os
import numpy as np

import mindspore as ms
from mindspore import mint, Tensor
import mindspore.common.dtype as mstype
from mindspore.communication import init

from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.inference.tensor_parallel.layers import (
    QKVParallelLinear,
    RowParallelLinear
)
from mindformers.parallel_core.inference.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.inference.transformer.dot_product_attention import DotProductAttention
from mindformers.parallel_core.inference.transformer.attention import (
    SelfAttention,
    SelfAttentionSubmodules
)
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_attention.data_gen_utils import (
    get_init_params,
    BLOCK_SIZE,
    NUM_BLOCKS
)


class SelfAttnRunner:
    """Class to manage Self_Attn module"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.use_flash_attention = self.args.use_flash_attention
        self.batch_size = self.args.batch_size
        self.seq_length = self.args.seq_length
        self.hidden_size = self.args.hidden_size
        self.is_prefill = self.args.is_prefill
        self.tensor_parallel = self.args.tensor_parallel

        init_params = get_init_params(self.is_prefill)

        self.input = ms.Tensor(init_params.get("input_sa"), dtype=mstype.bfloat16)

        if self.is_prefill:
            self.slot_mapping = Tensor(np.arange(self.batch_size * self.seq_length), mstype.int32)
            self.batch_valid_length = Tensor(np.ones((self.seq_length,)), dtype=mstype.int32)
        else:
            self.slot_mapping = Tensor(np.arange(self.batch_size * 1), mstype.int32)
            self.batch_valid_length = Tensor(np.ones((1,)), dtype=mstype.int32)

        self.block_tables = Tensor(np.ones((self.batch_size, BLOCK_SIZE)) * -1, mstype.int32)
        self.block_tables[0][0] = 0

        # RANK_ID and worker_num are set by msrun environment
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        # Set parallel context
        if self.rank_id is not None:
            init()
            initialize_model_parallel(tensor_model_parallel_size=self.tensor_parallel)

        self.config = self.get_self_attention_config()

    def get_self_attention_config(self):
        """Class to create Self_Attn config"""
        config = TransformerConfig(
            num_attention_heads=self.args.num_heads,
            num_query_groups=self.args.num_query_groups,
            hidden_size=self.hidden_size,
            sequence_parallel=False,
            num_layers=1,
            seq_length=self.seq_length,
            compute_dtype='bf16',
            layernorm_compute_dtype='bf16',
            add_qkv_bias=False,
            use_flash_attention=self.use_flash_attention,
            tensor_model_parallel_size=self.tensor_parallel,
            softmax_compute_dtype='bf16'
        )
        config.num_blocks = NUM_BLOCKS
        config.block_size = BLOCK_SIZE

        return config

    @staticmethod
    def _get_self_attn_spec(use_flash_attention):
        """Construct test self_attn spec."""

        if use_flash_attention:
            self_attn = ModuleSpec(module=SelfAttention,
                                   submodules=SelfAttentionSubmodules(
                                       core_attention=FlashAttention,
                                       linear_proj=RowParallelLinear,
                                       linear_qkv=QKVParallelLinear,
                                       q_layernorm=None,
                                       k_layernorm=None
                                   ))
        else:
            self_attn = ModuleSpec(module=SelfAttention,
                                   submodules=SelfAttentionSubmodules(
                                       core_attention=DotProductAttention,
                                       linear_proj=RowParallelLinear,
                                       linear_qkv=QKVParallelLinear,
                                       q_layernorm=None,
                                       k_layernorm=None
                                   ))
        return self_attn


    def build_model(self):
        """Build Self_Attention"""
        net = build_module(
            self._get_self_attn_spec(self.use_flash_attention),
            config=self.config,
            attn_mask_type=None,
            layer_number=1
        )
        return net

    def run(self):
        """Run self_attn with given inputs"""
        net = self.build_model()

        kv_cache_shape = (
            NUM_BLOCKS,
            BLOCK_SIZE,
            self.config.num_query_groups // self.config.tensor_model_parallel_size,
            self.config.hidden_size // self.config.num_attention_heads)
        key_cache = mint.zeros(kv_cache_shape, dtype=self.config.compute_dtype)
        value_cache = mint.zeros(kv_cache_shape, dtype=self.config.compute_dtype)

        if self.is_prefill:
            output = net(self.input,
                         attention_mask=None,
                         slot_mapping=self.slot_mapping,
                         actual_seq_qlen=self.batch_valid_length,
                         actual_seq_kvlen=self.batch_valid_length,
                         key_cache=key_cache,
                         value_cache=value_cache
                         )
        else:
            net.phase = "increment"
            net.add_flags(is_prefill=False)
            if self.use_flash_attention:
                net.core_attention.add_flags(is_prefill=False)

            output = net(self.input,
                         attention_mask=None,
                         block_tables=self.block_tables,
                         slot_mapping=self.slot_mapping,
                         batch_valid_length=self.batch_valid_length,
                         context_lens_tensor=self.batch_valid_length,
                         key_cache=key_cache,
                         value_cache=value_cache
                         )

        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.astype(mstype.float16).asnumpy() for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)

def main():
    parser = argparse.ArgumentParser(description="Run Self_Attn test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_query_groups", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--use_flash_attention", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--is_prefill", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--output_path", type=str, default="output_ms_sa.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)

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

    runner = SelfAttnRunner(args)
    # Execute Runner
    runner.run()


if __name__ == "__main__":
    main()
