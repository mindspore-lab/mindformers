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
"""Run SelfAttention accuracy test with configurable parameters via args"""
import argparse
import os
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore.communication import init
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.norm import LayerNorm
from mindformers.parallel_core.training_graph.transformer.attention import SelfAttention, \
    SelfAttentionSubmodules
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention

from data_gen_utils import get_init_params

SCRIPT_DIR = Path(__file__).parent.resolve()


class SelfAttentionMegatronRunner:
    """Class to manage SelfAttention model and weights"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.seq_len = self.args.seq_len
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.kv_hidden_size = self.args.kv_hidden_size
        self.num_attention_heads = self.args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.has_bias = self.args.has_bias
        self.compute_dtype = ms.bfloat16
        self.param_init_dtype = ms.float32

        # 获取初始参数
        init_params = get_init_params(
            self.seq_len, self.batch_size, self.hidden_size, self.head_dim * self.args.num_query_groups)

        self.hidden_states = ms.Tensor(init_params["hidden_states"], dtype=self.compute_dtype)
        self.attention_mask = ms.Tensor(init_params["attention_mask"], dtype=self.compute_dtype)
        self.weight_qkv = init_params["weight_qkv"]
        self.bias_qkv = init_params["bias_qkv"]
        self.weight_proj = init_params["weight_proj"]
        self.bias_proj = init_params["bias_proj"]
        self.q_layernorm_weight = init_params["q_layernorm_weight"]
        self.q_layernorm_bias = init_params["q_layernorm_bias"]
        self.k_layernorm_weight = init_params["k_layernorm_weight"]
        self.k_layernorm_bias = init_params["k_layernorm_bias"]

        # 设置并行环境
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None
        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        if self.rank_id is not None:
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True
            )
            init()

        # Transformer 配置
        self.config = TransformerConfig(
            compute_dtype='bfloat16',
            use_flash_attention=self.args.use_flash_attn,
            num_query_groups=self.args.num_query_groups,
            data_parallel_size=self.worker_num // self.args.tensor_parallel,
            tensor_model_parallel_size=self.args.tensor_parallel,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            add_bias_linear=self.has_bias,
            add_qkv_bias=self.has_bias,
            num_layers=1,
            params_dtype='float32',
            attention_dropout=0.0,
        )

    def build_model(self):
        """Build and initialize SelfAttention model"""
        submodules = SelfAttentionSubmodules(
            linear_qkv=ColumnParallelLinear,
            core_attention=FlashAttention,
            # core_attention=FlashAttention if self.config.use_flash_attn else CoreAttention,
            linear_proj=RowParallelLinear,
        )

        if self.args.q_layernorm == "Norm":
            submodules.q_layernorm = LayerNorm
        if self.args.k_layernorm == "Norm":
            submodules.k_layernorm = LayerNorm

        net = SelfAttention(
            config=self.config,
            submodules=submodules,
            layer_number=1,
        )

        state_dict = {}
        if self.weight_qkv is not None:
            state_dict["linear_qkv.weight"] = ms.Parameter(self.weight_qkv)
        if self.has_bias and self.bias_qkv is not None:
            state_dict["linear_qkv.bias"] = ms.Parameter(self.bias_qkv)

        if self.weight_proj is not None:
            state_dict["linear_proj.weight"] = ms.Parameter(self.weight_proj)
        if self.has_bias and self.bias_proj is not None:
            state_dict["linear_proj.bias"] = ms.Parameter(self.bias_proj)

        if self.args.q_layernorm == "Norm" and self.q_layernorm_weight is not None:
            state_dict["q_layernorm.weight"] = ms.Parameter(self.q_layernorm_weight)
            state_dict["q_layernorm.bias"] = ms.Parameter(self.q_layernorm_bias)
        if self.args.k_layernorm == "Norm" and self.k_layernorm_weight is not None:
            state_dict["k_layernorm.weight"] = ms.Parameter(self.k_layernorm_weight)
            state_dict["k_layernorm.bias"] = ms.Parameter(self.k_layernorm_bias)
        ms.load_param_into_net(net, state_dict, strict_load=True)
        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()
        output, bias = net(self.hidden_states, self.attention_mask)
        output_ms = {"output": output, "bias": bias}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.asnumpy().astype(np.float32) for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run SelfAttention test")
    parser.add_argument("--seq_len", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--kv_hidden_size", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--has_bias", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--use_flash_attn", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--num_query_groups", type=int, default=2)
    parser.add_argument("--q_layernorm", type=str, default=None)
    parser.add_argument("--k_layernorm", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="output_ms.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)

    args = parser.parse_args()

    ms.context.set_context(deterministic="ON")
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_seed(42)

    runner = SelfAttentionMegatronRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
