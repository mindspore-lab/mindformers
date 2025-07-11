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
"""Run mcore RoPE UT of inference with configurable parameters via args"""
import argparse
import os
from pathlib import Path
import numpy as np

import mindspore as ms
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.communication import init

from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.parallel_core.inference.base_models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
    Llama3RotaryEmbedding,
)
from mindformers.parallel_core.inference.base_models.common.embeddings.yarn_rotary_pos_embedding import \
    YaRNScalingRotaryEmbedding

from tests.st.test_ut.test_parallel_core.test_inference.test_common.test_embeddings.test_rotary_pos_embedding.data_gen_utils import (
    get_init_params,
    KV_CHANNELS,
)

SCRIPT_DIR = Path(__file__).parent.resolve()


class RotaryEmbeddingRunner:
    """Class to manage RoPE module"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.batch_size = self.args.batch_size
        self.seq_length = self.args.seq_length
        self.rotary_percent = self.args.rotary_percent
        self.rotary_interleaved = self.args.rotary_interleaved
        self.seq_len_interpolation_factor = self.args.seq_len_interpolation_factor
        self.rotary_base = self.args.rotary_base
        self.rotary_cos_format = self.args.rotary_cos_format
        self.max_position_embedding = self.args.max_position_embedding
        self.is_prefill = self.args.is_prefill
        self.rotary_dtype = mstype.bfloat16

        init_params = get_init_params(self.batch_size, self.seq_length, KV_CHANNELS)

        if self.is_prefill:
            self.query = Tensor(init_params.get("query_for_prefill"), dtype=mstype.bfloat16)
            self.key = Tensor(init_params.get("key_for_prefill"), dtype=mstype.bfloat16)
            self.batch_valid_length = Tensor(np.ones((self.batch_size,)), mstype.int32) * self.seq_length
        else:
            self.query = Tensor(init_params.get("query_for_decode"), dtype=mstype.bfloat16)
            self.key = Tensor(init_params.get("key_for_decode"), dtype=mstype.bfloat16)
            self.batch_valid_length = Tensor(np.ones((self.batch_size,)), mstype.int32)

        # RANK_ID and worker_num are set by msrun environment
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        # Set parallel context
        if self.rank_id is not None:
            init()
            initialize_model_parallel(tensor_model_parallel_size=1)


    def build_model(self):
        """Build RoPE module"""
        net = RotaryEmbedding(
            kv_channels=KV_CHANNELS,
            rotary_percent=self.rotary_percent,
            rotary_interleaved=self.rotary_interleaved,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            rotary_base=self.rotary_base,
            rotary_cos_format=self.rotary_cos_format,
            rotary_dtype=self.rotary_dtype,
            max_position_embeddings=self.max_position_embedding,
        )
        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()
        if self.is_prefill:
            rotary_pos_cos, rotary_pos_sin = net.get_cos_sin_for_prefill()
        else:
            rotary_pos_cos, rotary_pos_sin = net.get_cos_sin_for_decode(self.batch_valid_length)

        query, key = net(self.query, self.key, rotary_pos_cos, rotary_pos_sin, self.batch_valid_length)
        output_ms = {"query": query, "key": key}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.astype(mstype.float16).asnumpy() for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


class Llama3RotaryEmbeddingRunner(RotaryEmbeddingRunner):
    """Class to manage Llama3 RoPE module"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.scaling_factor = self.args.scaling_factor
        self.low_freq_factor = self.args.low_freq_factor
        self.high_freq_factor = self.args.high_freq_factor
        self.orig_max_position = self.args.orig_max_position
        super().__init__(args_from_parser)

    def build_model(self):
        """Build Llama3 RoPE module"""
        net = Llama3RotaryEmbedding(
            kv_channels=KV_CHANNELS,
            rotary_percent=self.rotary_percent,
            rotary_interleaved=self.rotary_interleaved,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            rotary_base=self.rotary_base,
            rotary_cos_format=self.rotary_cos_format,
            rotary_dtype=self.rotary_dtype,
            max_position_embeddings=self.max_position_embedding,
            scaling_factor=self.scaling_factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            orig_max_position=self.orig_max_position
        )
        return net


class YarnScalingRotaryEmbeddingRunner(RotaryEmbeddingRunner):
    """Class to manage Yarn Scaling RoPE module"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.scaling_factor = self.args.scaling_factor
        self.original_max_position_embeddings = self.args.orig_max_position
        self.beta_fast = self.args.beta_fast
        self.beta_slow = self.args.beta_slow
        self.mscale = self.args.mscale
        self.mscale_all_dim = self.args.mscale_all_dim
        super().__init__(args_from_parser)

    def build_model(self):
        """Build Llama3 RoPE module"""
        net = YaRNScalingRotaryEmbedding(
            kv_channels=KV_CHANNELS,
            rotary_percent=self.rotary_percent,
            rotary_interleaved=self.rotary_interleaved,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            rotary_base=self.rotary_base,
            rotary_cos_format=self.rotary_cos_format,
            rotary_dtype=self.rotary_dtype,
            scaling_factor=self.scaling_factor,
            original_max_position_embeddings=self.original_max_position_embeddings,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            mscale=self.mscale,
            mscale_all_dim=self.mscale_all_dim
        )
        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()
        if self.is_prefill:
            rotary_pos_cos, rotary_pos_sin = net.get_cos_sin_for_prefill()
        else:
            rotary_pos_cos, rotary_pos_sin = net.get_cos_sin_for_decode(self.batch_valid_length)

        query, key = net(self.query, self.key, rotary_pos_cos, rotary_pos_sin, self.batch_valid_length)
        output_ms = {"query": query, "key": key}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.astype(mstype.float16).asnumpy() for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run RoPE test")
    parser.add_argument("--module", type=str, default="default")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=1)
    parser.add_argument("--rotary_percent", type=float, default=1.0)
    parser.add_argument("--rotary_interleaved", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--seq_len_interpolation_factor", type=float, default=None)
    parser.add_argument("--rotary_base", type=float, default=10000.0)
    parser.add_argument("--rotary_cos_format", type=int, default=2)
    parser.add_argument("--max_position_embedding", type=int, default=1024)
    parser.add_argument("--is_prefill", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--scaling_factor", type=float, default=None)
    parser.add_argument("--low_freq_factor", type=float, default=None)
    parser.add_argument("--high_freq_factor", type=float, default=None)
    parser.add_argument("--orig_max_position", type=int, default=None)
    parser.add_argument("--beta_fast", type=int, default=None)
    parser.add_argument("--beta_slow", type=int, default=None)
    parser.add_argument("--mscale", type=float, default=None)
    parser.add_argument("--mscale_all_dim", type=float, default=None)
    parser.add_argument("--output_path", type=str, default="output_ms_rope.npz")

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

    # Init Runner to prepare inputs
    if args.module == "default":
        runner = RotaryEmbeddingRunner(args)
    elif args.module == "llama3":
        runner = Llama3RotaryEmbeddingRunner(args)
    elif args.module == "yarn":
        runner = YarnScalingRotaryEmbeddingRunner(args)
    else:
        raise ValueError(f"The RoPE unit test currently supports only "
                         f"RotaryEmbedding, Llama3RotaryEmbedding, and YaRNScalingRotaryEmbedding, "
                         f"but got {args.module}")
    # Execute Runner
    runner.run()


if __name__ == "__main__":
    main()
