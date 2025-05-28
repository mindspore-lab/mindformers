#  Copyright 2025 Huawei Technologies Co., Ltd
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""Run Multi-head Latent Attention (MLA) accuracy test with configurable parameters via args."""
import os
import argparse
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore.communication import init
from mindspore.mint import ones
from mindformers.parallel_core.training_graph.transformer.multi_latent_attention import MLASelfAttention, \
    MLASelfAttentionMegatron, MLASelfAttentionSubmodules, MLASelfAttentionSubmodulesMegatron
from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.training_graph.transformer.identity_op import IdentityOp
from mindformers.parallel_core.training_graph.transformer.norm import RMSNorm
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.parallel_core.utils.spec_utils import ModuleSpec, build_module
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_multi_latent_attention.data_gen_utils import get_init_params

SCRIPT_DIR = Path(__file__).parent.resolve()


class MLARunner:
    """Class to manage Multi-head Latent Attention (MLA) execution."""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.compute_dtype = "bfloat16"

        rank_id_str: str | None = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None
        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        if self.rank_id is not None:
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                full_batch=True
            )
            init()

        self.config = MLATransformerConfig(
            use_flash_attention=self.args.use_flash_attn,
            hidden_size=self.args.hidden_size,
            data_parallel_size=self.worker_num // self.args.tensor_parallel,
            tensor_model_parallel_size=self.args.tensor_parallel,
            compute_dtype=self.compute_dtype,
            num_attention_heads=2,
            num_layers=1,
            max_position_embeddings=2,
            q_lora_rank=self.args.q_lora_rank,
            kv_lora_rank=4,
            v_head_dim=8,
            qk_head_dim=4,
            qk_pos_emb_head_dim=4,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            output_layer_init_method=None,
            add_bias_linear=False
        )

    def build_model(self):
        """Build and initialize Multi-head Latent Attention model."""
        core_attention = FlashAttention
        q_layernorm = IdentityOp if self.args.q_layernorm.lower() == 'none' else RMSNorm
        k_layernorm = IdentityOp if self.args.k_layernorm.lower() == 'none' else RMSNorm

        if self.args.struct == 'a2':
            spec = ModuleSpec(
                module=MLASelfAttention,
                submodules=MLASelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    linear_qb=ColumnParallelLinear,
                    linear_kvb=ColumnParallelLinear,
                    core_attention=core_attention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=q_layernorm,
                    k_layernorm=k_layernorm
                )
            )
        else:
            spec = ModuleSpec(
                module=MLASelfAttentionMegatron,
                submodules=MLASelfAttentionSubmodulesMegatron(
                    linear_q_proj=ColumnParallelLinear,
                    linear_q_down_proj=ColumnParallelLinear,
                    linear_q_up_proj=ColumnParallelLinear,
                    linear_kv_down_proj=ColumnParallelLinear,
                    linear_kv_up_proj=ColumnParallelLinear,
                    core_attention=core_attention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=q_layernorm,
                    kv_layernorm=k_layernorm
                )
            )

        model = build_module(spec, config=self.config, layer_number=1)
        return model

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()
        hidden_state, mega_weight, mind_weight = get_init_params(0.0, 0.15, self.config, self.args.seq_length,
                                                                 self.args.batch_size, self.args.hidden_size)
        hidden_state = ms.Tensor(hidden_state, dtype=ms.float32)
        weight = mind_weight if self.args.struct == 'a2' else mega_weight
        for k in weight:
            weight[k] = ms.Parameter(ms.Tensor(weight[k], dtype=ms.float32))
        ms.load_param_into_net(net, weight)

        output = net(
            hidden_state,
            attention_mask=ones((self.args.batch_size, 1, self.args.seq_length, self.args.seq_length), dtype=ms.bool_),
            rotary_pos_emb=ones((4, 1, 1, 4))
        )
        output_ms = {"output": output}

        if self.rank_id is None or int(self.rank_id) == 0:
            # Convert to float32 for saving, common practice for bf16/fp16
            output_np = {k: v.asnumpy().astype(np.float32) for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Multi-head Latent Attention test")
    # Input shape parameters
    parser.add_argument("--struct", type=str, default="a2")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=16)
    # Model configuration parameters
    parser.add_argument("--q_lora_rank", type=int, default=8)
    parser.add_argument("--use_flash_attn", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--q_layernorm", type=str, default="RMSNorm")
    parser.add_argument("--k_layernorm", type=str, default="RMSNorm")
    # Output and parallelism
    parser.add_argument("--output_path", type=str, default="output_mla_ms.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--test_name", type=str, default="q16_flash_ql_kl")
    args = parser.parse_args()
    args.q_lora_rank = None if args.q_lora_rank == 0 else args.q_lora_rank

    ms.context.set_context(deterministic="ON")
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_seed(42)

    runner = MLARunner(args)
    runner.run()


if __name__ == "__main__":
    main()
