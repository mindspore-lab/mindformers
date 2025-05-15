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
"""
Test module for testing the column batched parallel linear used for mindformers.
"""
import argparse
import numpy as np

import mindspore as ms
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.graph.tensor_parallel.batched_layers import ColumnParallelBatchedLinear
from expected_output import get_output
from tests.utils.double_benchmark import DoubleBenchmarkComparator

ms.context.set_context(deterministic="ON")
ms.set_context(mode=ms.GRAPH_MODE)


def get_config():
    """get TransformerConfig for test"""
    return TransformerConfig(data_parallel=args.dp,
                             tensor_parallel=args.tp,
                             expert_model_parallel_size=args.ep,
                             num_attention_heads=4,
                             batch_size=1,
                             seq_length=4,
                             hidden_size=16,
                             ffn_hidden_size=16,
                             num_moe_experts=args.num_moe_experts,
                             compute_dtype=ms.bfloat16,
                             params_dtype=ms.float32,
                             )


def get_data(transformer_config: TransformerConfig):
    """get state_dict and input_tensor for test"""
    np.random.seed(1)
    weight_dict = {
        "linear.weight": np.random.rand(transformer_config.num_moe_experts, transformer_config.ffn_hidden_size,
                                        transformer_config.hidden_size),
    }
    for k in weight_dict:
        weight_dict[k] = ms.Parameter(ms.tensor(weight_dict[k], dtype=ms.float32) / 100)

    input_numpy = np.random.rand(transformer_config.seq_length, transformer_config.batch_size,
                                 transformer_config.hidden_size)
    input_tmp = np.tile(input_numpy, (1, 2, 1))
    input_tensor = ms.tensor(input_tmp, dtype=ms.float32)

    return weight_dict, input_tensor


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
    parser.add_argument(
        '--bias',
        action='store_true',
        help='linear has bias')
    parser.add_argument(
        '--skip_bias_add',
        action='store_true',
        help='linear skip bias add')
    parser.add_argument(
        '--skip_weight_param_allocation',
        action='store_true',
        help='linear skip weight param allocation')
    parser.add_argument(
        '--weight',
        action='store_true',
        help='the weight parameter in the linear forward pass is not None')

    parser.set_defaults(bias=False)
    parser.set_defaults(skip_bias_add=False)
    parser.set_defaults(skip_weight_param_allocation=False)
    parser.set_defaults(weight=False)
    args, rest_args = parser.parse_known_args()

    config = get_config()
    state_dict, input_ = get_data(config)
    net = ColumnParallelBatchedLinear(input_size=config.hidden_size, output_size=config.ffn_hidden_size,
                                      config=config, bias=args.bias, init_method=config.init_method_,
                                      skip_bias_add=args.skip_bias_add,
                                      skip_weight_param_allocation=args.skip_weight_param_allocation,
                                      compute_dtype=config.compute_dtype)
    if not args.skip_weight_param_allocation and not args.weight:
        ms.load_param_into_net(net, state_dict)
    if args.weight:
        output, bias = net(input_, state_dict['linear.weight'])
    else:
        output, bias = net(input_)
    gpu_output, golden_output = get_output()
    if args.bias and args.skip_bias_add:
        assert bias is not None
    else:
        assert bias is None
    npu_output = output.asnumpy()
    assert DoubleBenchmarkComparator.check_pass_or_not(npu_output, gpu_output, golden_output)

    print(f"Test Case Finished: dp:{args.dp}, tp:{args.tp}, ep:{args.ep}, num_moe_experts:{args.num_moe_experts}, "
          f"bias:{args.bias}, skip_bias_add:{args.skip_bias_add}, "
          f"skip_weight_param_allocation:{args.skip_weight_param_allocation}.")
