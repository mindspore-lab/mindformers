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
from mindformers.parallel_core.training_graph.transformer.attention import SelfAttentionContiguous, SelfAttentionSubmodules
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention
# from mindformers.parallel_core.training_graph.transformer.dot_product_attention import DotPruductAttention

from data_gen_utils import get_init_params

SCRIPT_DIR = Path(__file__).parent.resolve()


def transform_megatron_weight_to_mf(weight, kv_num_heads, n_rep, head_dim, tensor_parallel=1):
    """
    Transform the linear_qkv weight matrix from the commented code's grouped layout
    to the original code's global layout.

    Args:
        weight (np.ndarray): Weight matrix from commented code, shape (total_dim, hidden_dim)
        kv_num_heads (int): Number of key-value heads
        n_rep (int): Number of query head repetitions per key-value head
        head_dim (int): Dimension of each head

    Returns:
        np.ndarray: Transformed weight matrix, shape (total_dim, hidden_dim)
    """
    total_dim = weight.shape[0]
    expected_total_dim = head_dim * kv_num_heads * (
                n_rep + 2)  # hidden_size(head_dim * kv_num_heads * n_rep) + kv_hidden_size(head_dim * kv_num_heads) * 2
    assert total_dim == expected_total_dim, f"Total dimension mismatch: {total_dim} != {expected_total_dim}"

    # Total rows per group (query + key + value for one kv_head)
    group_rows = head_dim * (n_rep + 2)

    # Initialize lists to collect query, key, and value indices
    query_indices = []
    key_indices = []
    value_indices = []

    # Iterate over each kv_head group
    for kv_head in range(kv_num_heads):
        # Start index of this group
        group_start = kv_head * group_rows

        # Query rows: first n_rep * head_dim rows in the group
        query_start = group_start
        query_end = query_start + n_rep * head_dim
        query_indices.extend(range(query_start, query_end))

        # Key rows: next head_dim rows after query
        key_start = query_end
        key_end = key_start + head_dim
        key_indices.extend(range(key_start, key_end))

        # Value rows: last head_dim rows in the group
        value_start = key_end
        value_end = value_start + head_dim
        value_indices.extend(range(value_start, value_end))

    # Combine all indices in the order: all queries, all keys, all values
    if tensor_parallel == 1:
        new_indices = query_indices + key_indices + value_indices
    else:
        new_indices = []
        q_len = len(query_indices) // tensor_parallel
        k_len = len(key_indices) // tensor_parallel
        v_len = len(value_indices) // tensor_parallel
        for i in range(tensor_parallel):
            new_indices += query_indices[i * q_len:(i + 1) * q_len]
            new_indices += key_indices[i * k_len:(i + 1) * k_len]
            new_indices += value_indices[i * v_len:(i + 1) * v_len]

    new_indices = np.array(new_indices)

    # Reorder the rows of weight according to new_indices
    weight_new = weight[new_indices]

    return weight_new


class SelfAttentionRunner:
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
        self.weight_q = None
        self.bias_q = None
        self.weight_k = None
        self.bias_k = None
        self.weight_v = None
        self.bias_v = None
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
            # core_attention=FlashAttention if self.config.use_flash_attn else CoreAttention,
            core_attention=FlashAttention,
            linear_proj=RowParallelLinear,
        )

        if self.args.q_layernorm == "Norm":
            submodules.q_layernorm = LayerNorm
        if self.args.k_layernorm == "Norm":
            submodules.k_layernorm = LayerNorm

        net = SelfAttentionContiguous(
            config=self.config,
            submodules=submodules,
            layer_number=1,
        )

        state_dict = {}
        if self.weight_qkv is not None:
            weight_qkv = transform_megatron_weight_to_mf(
                np.copy(self.weight_qkv),
                kv_num_heads=self.config.num_query_groups,
                n_rep=self.config.num_attention_heads // self.config.num_query_groups,
                head_dim=self.config.kv_channels,
                tensor_parallel=self.config.tensor_model_parallel_size,
            )

            state_dict["linear_qkv.weight"] = ms.Parameter(weight_qkv)
        if self.has_bias and self.bias_qkv is not None:
            bias_qkv = transform_megatron_weight_to_mf(
                np.copy(self.bias_qkv),
                kv_num_heads=self.config.num_query_groups,
                n_rep=self.config.num_attention_heads // self.config.num_query_groups,
                head_dim=self.config.kv_channels,
                tensor_parallel=self.config.tensor_model_parallel_size,
            )
            state_dict["linear_qkv.bias"] = ms.Parameter(bias_qkv)

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

    runner = SelfAttentionRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
