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
"""Run mcore MoE UT of inference with configurable parameters via args"""
import argparse
import os
from pathlib import Path
import numpy as np

import mindspore as ms
from mindspore import Parameter
import mindspore.common.dtype as mstype
from mindspore.communication import init, get_rank

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import build_module
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from mindformers.parallel_core.inference.base_models.gpt.moe_module_spec import get_moe_module_spec

from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_moe.data_gen_utils import (
    get_init_params,
)

SCRIPT_DIR = Path(__file__).parent.resolve()


class MoERunner:
    """Class to manage MoE module"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser

        self.seq_len = self.args.seq_len
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.num_experts = self.args.num_experts
        self.moe_intermediate_size = self.args.moe_intermediate_size
        self.moe_shared_expert_intermediate_size = self.args.moe_shared_expert_intermediate_size
        self.routed_scaling_factor = self.args.routed_scaling_factor
        self.n_shared_experts = self.args.n_shared_experts
        self.num_experts_per_tok = self.args.num_experts_per_tok
        self.n_group = self.args.n_group
        self.topk_group = self.args.topk_group

        init_params = get_init_params(self.seq_len, self.batch_size, self.hidden_size,
                                      self.num_experts, self.moe_intermediate_size)

        self.input = ms.Tensor(init_params.pop("input"), dtype=mstype.bfloat16)
        self.param_dict = init_params

        # RANK_ID and worker_num are set by msrun environment
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        # Set parallel context
        if self.rank_id is not None:
            init()
            initialize_model_parallel(tensor_model_parallel_size=self.args.tensor_parallel)

        # Transformer config
        self.config = TransformerConfig(
            num_layers=1,
            num_attention_heads=8,
            hidden_size=self.hidden_size,
            hidden_act="silu",
            num_moe_experts=self.num_experts,
            moe_router_topk=self.num_experts_per_tok,
            shared_expert_num=self.n_shared_experts,
            moe_router_topk_scaling_factor=self.routed_scaling_factor,
            moe_shared_expert_intermediate_size=self.moe_shared_expert_intermediate_size,
            moe_ffn_hidden_size=self.moe_intermediate_size,
            moe_router_group_topk=self.topk_group,
            moe_router_num_groups=self.n_group,
            add_bias_linear=False,
            gated_linear_unit=True,
            compute_dtype="bfloat16",
            params_dtype="bfloat16",
            moe_router_dtype="bfloat16",
            moe_router_enable_expert_bias=True,
            moe_router_score_function="sigmoid",
        )

    def build_model(self):
        """Build MLP module"""
        net = build_module(
            get_moe_module_spec(num_experts=self.config.num_moe_experts),
            config=self.config,
        )

        self._load_weights(net, self.param_dict)

        return net

    def _load_weights(self, net, param_dict):
        """load weights for moe module"""
        tp_group_size = self.args.tensor_parallel
        rank_id = get_rank()
        new_param_dict = {}

        def split(weight, split_axis=0):
            split_size = weight.shape[split_axis] // tp_group_size
            start = rank_id * split_size
            stop = (rank_id + 1) * split_size
            if split_axis == 0:
                return weight[start:stop]
            if split_axis == 1:
                return weight[:, start:stop]
            return weight[:, :, start:stop]

        # shared expert weight
        shared_expert_fc1_w = param_dict["shared_experts.linear_fc1.weight"]
        shared_expert_fc2_w = param_dict["shared_experts.linear_fc2.weight"]
        shared_expert_w_gate = shared_expert_fc1_w[:self.moe_intermediate_size, :]
        shared_expert_w_hidden = shared_expert_fc1_w[self.moe_intermediate_size:, :]
        shared_expert_w_gate_shard = split(shared_expert_w_gate)
        shared_expert_w_hidden_shard = split(shared_expert_w_hidden)
        shared_expert_w_fc1_shard = np.concatenate([shared_expert_w_gate_shard, shared_expert_w_hidden_shard], axis=0)
        shared_expert_w_fc2_shard = split(shared_expert_fc2_w, split_axis=1)
        print(f"hared_expert_w_fc1_shard: {shared_expert_w_fc1_shard.shape}")
        print(f"hared_expert_w_fc2_shard: {shared_expert_w_fc2_shard.shape}")
        new_param_dict["shared_experts.linear_fc1.weight"] = Parameter(shared_expert_w_fc1_shard)
        new_param_dict["shared_experts.linear_fc2.weight"] = Parameter(shared_expert_w_fc2_shard)

        # router
        router_weight = param_dict["router.weight.weight"]
        new_param_dict["router.weight.weight"] = Parameter(router_weight)
        router_expert_bias = param_dict["router.expert_bias"]
        new_param_dict["router.expert_bias"] = Parameter(router_expert_bias)

        # expert
        expert_fc1_w = param_dict["experts.weight1"]
        expert_fc2_w = param_dict["experts.weight2"]
        expert_w_gate = expert_fc1_w[:, :, :self.moe_intermediate_size]
        expert_w_hidden = expert_fc1_w[:, :, self.moe_intermediate_size:]
        expert_w_gate_shard = split(expert_w_gate, split_axis=2)
        expert_w_hidden_shard = split(expert_w_hidden, split_axis=2)
        print(f"expert_w_gate_shard: {expert_w_gate_shard.shape}")
        print(f"expert_w_hidden_shard: {expert_w_hidden_shard.shape}")
        expert_w_fc1_shard = np.concatenate([expert_w_gate_shard, expert_w_hidden_shard], axis=2)
        expert_w_fc2_shard = split(expert_fc2_w, split_axis=1)
        new_param_dict["experts.weight1"] = Parameter(expert_w_fc1_shard)
        new_param_dict["experts.weight2"] = Parameter(expert_w_fc2_shard)

        ms.load_param_into_net(net, new_param_dict)

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()

        output = net(self.input)
        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.astype(mstype.float16).asnumpy() for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run MLP test")
    parser.add_argument("--seq_len", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--n_shared_experts", type=int, default=1)
    parser.add_argument("--routed_scaling_factor", type=float, default=32)
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--n_group", type=int, default=2)
    parser.add_argument("--topk_group", type=int, default=2)
    parser.add_argument("--moe_intermediate_size", type=int, default=8)
    parser.add_argument("--moe_shared_expert_intermediate_size", type=int, default=8)
    parser.add_argument("--output_path", type=str, default="output_ms.npz")
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

    # Prepare input
    runner = MoERunner(args)
    runner.run()


if __name__ == "__main__":
    main()
