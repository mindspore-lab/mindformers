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
"""Run TransformerLayer accuracy test with configurable parameters via args"""

import argparse
import os
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore.communication import init

from mindformers.parallel_core.training_graph.transformer.identity_op import IdentityOp
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.transformer_layer import TransformerLayer, \
    TransformerLayerSubmodules
from mindformers.parallel_core.training_graph.transformer.norm import Norm
from mindformers.parallel_core.training_graph.transformer.attention import SelfAttentionMegatron, \
    SelfAttentionSubmodules
from mindformers.parallel_core.training_graph.transformer.mlp import MLP, MLPSubmodules
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention

from data_gen_utils import get_init_params, DEFAULT_SEQ_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_HIDDEN_SIZE, \
    DEFAULT_FFN_HIDDEN_SIZE, DEFAULT_NUM_HEADS

SCRIPT_DIR = Path(__file__).parent.resolve()

MODULE_MAP = {
    "IdentityOp": IdentityOp,
    "Norm": Norm,
    "SelfAttention": SelfAttentionMegatron,
    "MLP": MLP,
}


class TransformerLayerRunner:
    """Class to manage TransformerLayer model and execution"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser

        # Model dimensions
        self.hidden_size = self.args.hidden_size
        self.seq_length = self.args.seq_length
        self.batch_size = self.args.batch_size
        self.ffn_hidden_size = self.args.ffn_hidden_size
        self.num_attention_heads = self.args.num_attention_heads

        self.compute_dtype = ms.bfloat16
        self.param_init_dtype = ms.float32

        # Parallelism
        self.tensor_parallel = self.args.tensor_parallel
        self.rank_id = None
        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        rank_id_str = os.environ.get("RANK_ID")
        if rank_id_str is not None:
            self.rank_id = int(rank_id_str)

        if self.rank_id is not None:
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True,
                # device_num=self.worker_num # device_num is often set by environment
            )
            init()  # Initialize communication

        self.data_parallel = self.worker_num // self.tensor_parallel
        if self.worker_num % self.tensor_parallel != 0:
            raise ValueError(
                f"worker_num ({self.worker_num}) must be divisible by tensor_parallel ({self.tensor_parallel})"
            )

        # Transformer config
        self.config = TransformerConfig(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            num_attention_heads=self.num_attention_heads,
            seq_length=self.seq_length,
            data_parallel_size=self.data_parallel,
            tensor_model_parallel_size=self.tensor_parallel,
            compute_dtype='bf16',
            layernorm_compute_dtype='fp32',
            normalization="LayerNorm",
            num_layers=1,
            params_dtype='fp32'
        )

        # Submodules
        self.submodules_spec = TransformerLayerSubmodules(
            input_layernorm=Norm,
            self_attention=ModuleSpec(
                module=MODULE_MAP[self.args.self_attention],
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=FlashAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            pre_cross_attn_layernorm=MODULE_MAP[self.args.pre_cross_attn_layernorm],
            cross_attention=MODULE_MAP[self.args.cross_attention],
            pre_mlp_layernorm=MODULE_MAP[self.args.pre_mlp_layernorm],
            mlp=ModuleSpec(
                module=MODULE_MAP[self.args.mlp],
                submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear)
            )
            # self_attn_bda and other BDA modules are IdentityOp by default in TransformerLayerSubmodules
        )

        # Inputs
        init_input_params = get_init_params(
            seq_length=self.seq_length,
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            compute_dtype=self.compute_dtype
        )
        self.hidden_states = init_input_params.get("hidden_states")
        self.attention_mask = init_input_params.get("attention_mask")

    def build_model(self):
        """Build and initialize TransformerLayer model"""
        net = TransformerLayer(
            config=self.config,
            submodules=self.submodules_spec,
            layer_number=1
        )

        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()
        net.set_train(False)  # Set to eval mode

        output, context, extra_loss = net(
            self.hidden_states,
            attention_mask=self.attention_mask,
            extra_loss=ms.Tensor(0.0, dtype=self.compute_dtype)  # Initial extra_loss
        )

        output_ms = {"output": output}
        if extra_loss is not None:  # MoE or other layers might return extra_loss
            output_ms["extra_loss"] = extra_loss
        if context is not None:  # If cross attention was used and returned context
            output_ms["context"] = context

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {}
            for k, v_tensor in output_ms.items():
                if v_tensor is not None:
                    # Ensure tensor is moved to CPU before converting to NumPy if it's on device
                    # For bfloat16, convert to float32 for saving with NumPy, as NumPy might not support bfloat16 directly.
                    if v_tensor.dtype == ms.bfloat16:
                        output_np[k] = v_tensor.to(ms.float32).asnumpy()
                    else:
                        output_np[k] = v_tensor.asnumpy()

            output_path = self.args.output_path
            np.savez(output_path, **output_np)
            print(f"Output saved to {output_path}")
            for k, v in output_np.items():
                print(f"Saved output key '{k}' with shape {v.shape} and dtype {v.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Run TransformerLayer test")
    # Model dimensions
    parser.add_argument("--seq_length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--ffn_hidden_size", type=int, default=DEFAULT_FFN_HIDDEN_SIZE)
    parser.add_argument("--num_attention_heads", type=int, default=DEFAULT_NUM_HEADS)

    # Submodule types (must match keys in MODULE_MAP)
    parser.add_argument("--input_layernorm", type=str, default="Norm", choices=MODULE_MAP.keys())
    parser.add_argument("--self_attention", type=str, default="SelfAttention", choices=MODULE_MAP.keys())
    parser.add_argument("--pre_cross_attn_layernorm", type=str, default="IdentityOp", choices=MODULE_MAP.keys())
    parser.add_argument("--cross_attention", type=str, default="IdentityOp", choices=MODULE_MAP.keys())
    parser.add_argument("--pre_mlp_layernorm", type=str, default="Norm", choices=MODULE_MAP.keys())
    parser.add_argument("--mlp", type=str, default="MLP", choices=MODULE_MAP.keys())

    # Execution config
    parser.add_argument("--output_path", type=str, default="output_transformer_layer.npz")
    parser.add_argument("--tensor_parallel", type=int, default=1)

    args = parser.parse_args()

    ms.context.set_context(deterministic="ON")
    ms.set_context(mode=ms.GRAPH_MODE)  # GRAPH_MODE is typical for MindSpore model execution
    ms.set_seed(42)
    np.random.seed(42)

    runner = TransformerLayerRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
