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
"""Test module for testing RowParallelBatchedLinear used for mindformers."""
import argparse

import mindspore as ms
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.tensor_parallel.batched_layers import RowParallelBatchedLinear
from data_gen_utils import get_data, get_output
from tests.utils.double_benchmark import DoubleBenchmarkComparator, DoubleBenchmarkStandard

ms.context.set_context(deterministic="ON")
ms.set_context(mode=ms.GRAPH_MODE)


def get_config():
    """get TransformerConfig for test"""
    return TransformerConfig(data_parallel_size=args.dp,
                             tensor_model_parallel_size=args.tp,
                             expert_model_parallel_size=args.ep,
                             num_layers=1,
                             num_attention_heads=4,
                             hidden_size=16,
                             ffn_hidden_size=16,
                             num_moe_experts=args.num_moe_experts,
                             compute_dtype='bfloat16',
                             params_dtype='float32',
                             add_bias_linear=False
                             )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dp',
        default=1,
        required=False,
        type=int,
        help='data_parallel')
    parser.add_argument(
        '--tp',
        default=1,
        required=False,
        type=int,
        help='tensor_parallel')
    parser.add_argument(
        '--ep',
        default=1,
        required=False,
        type=int,
        help='expert_model_parallel')
    parser.add_argument(
        '--num_moe_experts',
        default=2,
        required=False,
        type=int,
        help='num_moe_experts')

    args, rest_args = parser.parse_known_args()
    config = get_config()
    state_dict, input_ = get_data(config)
    net = RowParallelBatchedLinear(input_size=config.hidden_size, output_size=config.ffn_hidden_size,
                                   config=config, init_method=config.init_method)
    ms.load_param_into_net(net, state_dict)
    output, bias = net(input_)
    gpu_output, golden_output = get_output()
    npu_output = output.asnumpy()
    standard = DoubleBenchmarkStandard(dtype="bfloat16")
    assert DoubleBenchmarkComparator.check_pass_or_not(npu_output, gpu_output, golden_output, standard)
