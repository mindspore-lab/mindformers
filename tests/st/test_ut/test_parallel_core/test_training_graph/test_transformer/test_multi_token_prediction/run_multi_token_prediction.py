#  Copyright 2025 Huawei Technologies Co., Ltd
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""Run Multi-Token Prediction (MTP) accuracy test with configurable parameters via args."""
import os
import argparse
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter, set_auto_parallel_context, load_param_into_net, set_context, set_seed
from mindspore.communication import init

from mindformers.parallel_core.training_graph.transformer.multi_token_prediction import MultiTokenPredictionLayer, \
    MultiTokenPredictionLayerSubmodules, MultiTokenPredictionBlock, MultiTokenPredictionBlockSubmodules
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.attention import SelfAttentionMegatron, \
    SelfAttentionSubmodules
from mindformers.parallel_core.training_graph.transformer.transformer_layer import TransformerLayer, \
    TransformerLayerSubmodules
from mindformers.parallel_core.training_graph.transformer.identity_op import IdentityOp
from mindformers.parallel_core.training_graph.transformer.norm import RMSNorm
from mindformers.parallel_core.training_graph.transformer.mlp import MLP, MLPSubmodules
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_multi_token_prediction.data_gen_utils import \
    get_init_params

SCRIPT_DIR = Path(__file__).parent.resolve()


class MTPRunner:
    """Class to manage Multi-Token Prediction (MTP) execution."""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.compute_dtype = ms.bfloat16

        rank_id_str: str | None = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None
        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        if self.rank_id is not None:
            set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                full_batch=True
            )
            init()

        self.config = TransformerConfig(
            hidden_size=self.args.hidden_size,
            seq_length=self.args.seq_length,
            data_parallel_size=self.worker_num // self.args.tp,
            tensor_model_parallel_size=self.args.tp,
            compute_dtype=self.compute_dtype,
            position_embedding_type=self.args.position_embedding_type,
            num_attention_heads=2,
            num_layers=1,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            output_layer_init_method=None,
            add_bias_linear=False,
            calculate_per_token_loss=False,
            mtp_loss_scaling_factor=0.5,
            mtp_num_layers=2
        )

    def build_model(self):
        """Build and initialize Multi-Token Prediction (MTP) model."""
        transfmr_spec = ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=RMSNorm,
                self_attention=ModuleSpec(
                    module=SelfAttentionMegatron,
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=ColumnParallelLinear,
                        core_attention=FlashAttention,
                        linear_proj=RowParallelLinear,
                    ),
                ),
                pre_cross_attn_layernorm=IdentityOp,
                cross_attention=IdentityOp,
                pre_mlp_layernorm=RMSNorm,
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear)
                )
            )
        )
        mtp_layer_spec = ModuleSpec(
            module=MultiTokenPredictionLayer,
            submodules=MultiTokenPredictionLayerSubmodules(
                enorm=RMSNorm,
                hnorm=RMSNorm,
                eh_proj=ColumnParallelLinear,
                transformer_layer=transfmr_spec,
                layer_norm=RMSNorm
            )
        )
        mtp_block_spec = ModuleSpec(
            module=MultiTokenPredictionBlock,
            submodules=MultiTokenPredictionBlockSubmodules(
                layer_specs=[mtp_layer_spec for _ in range(self.config.mtp_num_layers)]
            )
        )

        mtp_block = MultiTokenPredictionBlock(self.config, mtp_block_spec, self.args.vocab_size)

        return mtp_block

    def run(self):
        """Run the model with given inputs"""
        mtp_block = self.build_model()

        params = get_init_params(
            self.config,
            0.0,
            0.15,
            self.args.seq_length,
            self.args.batch_size,
            self.args.vocab_size
        )

        # Load params into net
        state_dict = params['state_dict']
        for k in state_dict:
            state_dict[k] = Parameter(Tensor(state_dict[k], dtype=ms.bfloat16), name=k)
        load_param_into_net(mtp_block, state_dict)

        output = mtp_block(
            Tensor(params['input_ids'], dtype=ms.int32),
            Tensor(params['position_ids'], dtype=ms.int32),
            Tensor(params['hidden_states'], dtype=ms.bfloat16),
            Tensor(params['attention_mask'], dtype=ms.bool_),
            labels=Tensor(params['labels'], dtype=ms.int32),
            loss_mask=Tensor(params['loss_mask'], dtype=ms.bool_),
            rotary_pos_emb=None,
            word_embeddings_weight=state_dict['word_embeddings.weight'],
            position_embeddings_weight=state_dict['position_embeddings.weight'],
            output_weight=state_dict['weight'],
        )
        print(output)
        if self.rank_id is None or int(self.rank_id) == 0:
            # Convert to float32 for saving, common practice for bf16/fp16
            output_np = {
                'mtp_loss': output[0].asnumpy().astype(np.float32),
                'extra_loss': output[1].asnumpy().astype(np.float32)
            }
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Multi-Token Prediction (MTP) test")
    # Input shape parameters
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=100)
    # Model Args
    parser.add_argument("--position_embedding_type", type=str, default="learned_absolute")
    # Output and parallelism
    parser.add_argument("--test_name", type=str, default="single_card_baseline")
    parser.add_argument("--output_path", type=str, default="output_mtp_ms.npz")
    parser.add_argument("--dp", type=int, default=1, help='data_parallel')
    parser.add_argument("--cp", type=int, default=1, help='context_parallel')
    parser.add_argument("--tp", type=int, default=1, help='tensor_parallel')
    args = parser.parse_args()

    set_context(deterministic="ON")
    set_context(mode=ms.GRAPH_MODE)
    set_seed(42)

    runner = MTPRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
