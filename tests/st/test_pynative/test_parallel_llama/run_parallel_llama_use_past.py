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
from itertools import chain

import mindspore as ms
import mindspore.common.dtype as mstype
import numpy as np
from mindspore.communication import get_rank
from mindspore.communication.management import init

from mindformers import MindFormerConfig
from mindformers.experimental.distri_cores.create_comm import \
    initialize_model_parallel
from mindformers.experimental.llama_demo import ParallelLlamaForCausalLM
from mindformers.models.llama import LlamaConfig
from mindformers.modules.block_tables import BlockTables
from mindformers.modules.transformer.transformer import TransformerOpParallelConfig
from tests.st.test_pynative.test_parallel_llama.run_parallel_llama import convert_model_config


def get_valid_length_each_example(input_ids, pad_token_id):
    """get valid length and max length in a batch"""
    batch_size = input_ids.shape[0]
    valid_length_each_example = []
    for i in range(batch_size):
        # As the nonzero returns the index and we need length
        valid_length_each_example.append(
            np.max(np.argwhere(input_ids[i] != pad_token_id))
            + 1
        )
    valid_length_each_example = np.array(valid_length_each_example)
    max_length = np.max(valid_length_each_example)
    return batch_size, valid_length_each_example, max_length


def predict_parallel_llama(configs):
    """Test ParallelTransformer."""

    parallel_config = configs.parallel_config
    ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE, deterministic="ON",
                   jit_config={"jit_level": "O0", "infer_boost": "on"})

    init()
    initialize_model_parallel(tp_size=parallel_config.tensor_parallel)

    ms.set_seed(2024)

    network = ParallelLlamaForCausalLM(configs)
    for name, param in network.parameters_and_names():
        print(f"{name} {param.dtype} {param.shape}")

    graph_ckpt = ms.load_checkpoint(f"{configs.load_checkpoint}/rank_{get_rank()}/checkpoint_{get_rank()}.ckpt")
    ms.load_param_into_net(network, graph_ckpt)

    network.set_train(False)
    expect_target_ids = [
        [1, 306, 5360, 1522, 823, 292, 29892, 1363, 28054, 20659, 18196, 27342, 2153, 9026, 18065, 3946],
        [1, 15043, 27993, 28102, 20426, 4798, 11285, 19046, 29265, 936, 537, 4798, 11285, 1306, 550, 4034]]
    input_ids = [[1, 306, 5360, 1522, 823, 292, 29892, 1363, 0, 0, 0, 0, 0, 0, 0],
                 [1, 15043, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    block_mgr = BlockTables(configs.num_blocks, configs.block_size, configs.seq_length)

    prefill = True
    inputs_ids = np.array(input_ids)
    batch_size, batch_valid_length, batch_max_length = get_valid_length_each_example(inputs_ids,
                                                                                     configs.pad_token_id)
    block_mgr.init_cache_engine(batch_size)
    is_finished = [False] * batch_size
    original_ids = [input_ids[i][:batch_valid_length[i]] for i in range(batch_size)]
    while np.sum(is_finished) != batch_size:
        if prefill:
            network.add_flags_recursive(is_first_iteration=True)
            block_tables, slot_mapping = block_mgr.assemble_pa_full_inputs(batch_max_length, batch_valid_length,
                                                                           is_finished)
            prefill_input_ids = inputs_ids[:, :batch_max_length]
            data = {"input_ids": ms.Tensor(prefill_input_ids, mstype.int32).reshape(-1, batch_max_length.item()),
                    "block_tables": ms.Tensor(block_tables, mstype.int32),
                    "slot_mapping": ms.Tensor(slot_mapping, mstype.int32),
                    "batch_valid_length": ms.Tensor(batch_valid_length, mstype.int32).reshape(-1, )}
        else:
            block_tables, slot_mapping = block_mgr.assemble_pa_inc_inputs(batch_valid_length,
                                                                          is_finished)
            data = {"input_ids": ms.Tensor(target_list, mstype.int32).reshape(-1, 1),
                    "block_tables": ms.Tensor(block_tables, mstype.int32),
                    "slot_mapping": ms.Tensor(slot_mapping, mstype.int32),
                    "batch_valid_length": ms.Tensor(batch_valid_length, mstype.int32).reshape(-1, )}
        logits = network(**data)
        if isinstance(logits, tuple):
            logits = logits[0]
        target_list = ms.ops.argmax(logits, -1)
        target_list = target_list.asnumpy().tolist()
        if isinstance(target_list[0], list):
            target_list = list(chain(*target_list))
        for i in range(batch_size):
            if is_finished[i]:
                continue
            batch_valid_length[i] += int(1)
            original_ids[i].append(target_list[i])
            if target_list[i] == model_config.eos_token_id or batch_valid_length[i] == model_config.seq_length:
                is_finished[i] = True

        network.add_flags_recursive(is_first_iteration=False)
        prefill = False
    assert expect_target_ids == original_ids, "the output ids is not same with expected ids."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="config yaml.")
    parser.add_argument("--checkpoint_path", type=str, help="checkpoint path")

    parallel_config = TransformerOpParallelConfig(
        model_parallel=2,
        vocab_emb_dp=False,
    )
    args, rest_args = parser.parse_known_args()
    config = MindFormerConfig(args.config_path)
    model_config = LlamaConfig(**config.model.model_config)
    model_config.num_layers = 1
    model_config.seq_length = 16
    model_config.parallel_config = parallel_config
    model_config.load_checkpoint = args.checkpoint_path
    model_config.checkpoint_name_or_path = None

    converted_llama_config = convert_model_config(model_config)
    predict_parallel_llama(converted_llama_config)
