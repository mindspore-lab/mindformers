# Copyright 2024 Huawei Technologies Co., Ltd
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
""" Test Pallel LLaMa. """
import argparse
import os

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from mindspore.communication import get_rank
from mindspore.communication.management import init

from mindformers.experimental.distri_cores.create_comm import \
    initialize_model_parallel
from mindformers.experimental.llama_demo import ParallelLlamaForCausalLM
from mindformers.models.llama import LlamaConfig, LlamaForCausalLM
from mindformers.modules.transformer.op_parallel_config import \
    default_dpmp_config
from mindformers.modules.transformer.transformer import (
    TransformerOpParallelConfig, default_transformer_recompute_config)


class GoldenNet(nn.Cell):
    """GoldenNet."""

    def __init__(self, configs: LlamaConfig):
        super().__init__()
        self.network = LlamaForCausalLM(configs)

    def construct(self, input_ids):
        output = self.network(input_ids=input_ids)
        return output


class ParallelNet(nn.Cell):
    def __init__(self, configs):
        super().__init__()
        self.network = ParallelLlamaForCausalLM(configs)

    def construct(self, input_ids):
        output = self.network(input_ids)
        return output


def print_data(obj, name):
    if obj is not None:
        print(f"{name}: {[obj]}")
        print(f"{name}.dtype: {obj.dtype}")
        print(f"{name}.shape: {obj.shape}")
        print(f"{name}.mean: {obj.mean()}")
        print(f"{name}.max: {obj.max()}")
        print(f"{name}.min: {obj.min()}")


def generate_golden(configs):
    """Generate golden."""

    vocab_size = configs.vocab_size
    batch_size = configs.batch_size
    seq_length = configs.seq_length

    ms.set_context(
        device_target="Ascend",
        mode=ms.GRAPH_MODE,
        deterministic="ON",
        jit_config={"jit_level": "O0"},
    )

    init()
    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(
        parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
        strategy_ckpt_save_file="str.ckpt",
    )

    ms.set_seed(2024)
    input_ids_np = np.random.randint(
        low=1, high=vocab_size, size=(batch_size, seq_length), dtype=np.int32
    )
    input_ids = Tensor(input_ids_np)

    default_dpmp_config.recompute = default_transformer_recompute_config
    network = GoldenNet(configs)
    network.set_train(False)
    for name, param in network.parameters_and_names():
        print(f"{name} {param.shape}")

    logits = network(input_ids)
    print_data(logits, "logits")
    ms.save_checkpoint(network, f"llama_golden_{get_rank()}.ckpt", False)
    np.save(f"golden_logits_{get_rank()}.npy", logits.numpy())


def run_parallel_llama(configs, mode):
    """Test ParallelTransformer."""
    batch_size = configs.batch_size

    seq_length = configs.seq_length
    vocab_size = configs.vocab_size

    ms.set_context(device_target="Ascend", mode=mode, deterministic="ON")

    init()
    initialize_model_parallel(tp_size=configs.parallel_config.model_parallel)

    ms.set_seed(2024)
    input_ids_np = np.random.randint(
        low=1, high=vocab_size, size=(batch_size, seq_length), dtype=np.int32
    )
    input_ids = Tensor(input_ids_np)

    network = ParallelNet(configs)
    for name, param in network.parameters_and_names():
        print(f"{name} {param.dtype} {param.shape}")

    graph_ckpt = ms.load_checkpoint(f"llama_golden_{get_rank()}.ckpt")
    ms.load_param_into_net(network, graph_ckpt)

    network.set_train(False)
    logits = network(input_ids)
    print_data(logits, "logits")
    golden_logits = np.load(f"golden_logits_{get_rank()}.npy")
    assert np.allclose(logits.numpy(), golden_logits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_golden", action="store_true", help="Generate golden data for test."
    )
    parser.add_argument(
        "--mode", choices=["graph", "pynative"], default="graph", help="Generate golden data for test."
    )

    args, rest_args = parser.parse_known_args()

    parallel_config = TransformerOpParallelConfig(
        model_parallel=2
    )

    llama_config = LlamaConfig(
        batch_size=1,
        seq_length=4,
        num_heads=8,
        hidden_size=16,
        num_layers=2,
        intermediate_size=64,
        compute_dtype=mstype.float16,
        param_init_type=mstype.float16,
        layernorm_compute_type=mstype.float16,
        softmax_compute_type=mstype.float16,
        rotary_dtype=mstype.float16,
        vocab_size=128,
        qkv_concat=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
    )

    os.environ["RUN_MODE"] = "predict"
    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE

    if args.generate_golden:
        generate_golden(llama_config)
    else:
        run_parallel_llama(llama_config, mode)
