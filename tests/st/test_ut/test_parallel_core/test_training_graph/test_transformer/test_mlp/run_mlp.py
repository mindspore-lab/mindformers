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
"""Test module for testing MLP used for mindformers."""
import argparse

import mindspore as ms
from mindspore import nn, ParameterTuple
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.mlp import MLP, MLPSubmodules
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from data_gen_utils import get_init_params, get_golden_datas, get_gpu_datas, get_grads
from tests.utils.double_benchmark import DoubleBenchmarkComparator, DoubleBenchmarkStandard
from tests.utils.tensor_utils import to_numpy_list

ms.context.set_context(deterministic="ON")
ms.set_context(mode=ms.GRAPH_MODE)


def get_config():
    """get TransformerConfig for test"""
    return TransformerConfig(num_layers=1,
                             num_attention_heads=2,
                             hidden_size=16,
                             ffn_hidden_size=16,
                             gated_linear_unit=args.gated_linear_unit,
                             add_bias_linear=args.add_bias_linear,
                             hidden_act="silu",
                             compute_dtype='bfloat16',
                             params_dtype='float32'
                             )


class TestModel(nn.Cell):
    """Model for test"""

    def __init__(self, config: TransformerConfig, input_size=None):
        super(TestModel, self).__init__()
        self.config = config
        self.mlp = MLP(submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
                       config=config, input_size=input_size)

    def construct(self, hidden_states):
        """This avoids graph compilation errors due to unsupported return types."""
        if self.config.add_bias_linear:
            mlp_output, mlp_bias, _ = self.mlp(hidden_states)
            return mlp_output, mlp_bias
        mlp_output = self.mlp(hidden_states)
        return mlp_output[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_size',
        default=None,
        required=False,
        type=int,
        help='hidden_size of linear')
    parser.add_argument(
        '--add_bias_linear',
        action='store_true',
        help='include a bias term in linear layers')
    parser.add_argument(
        '--gated_linear_unit',
        action='store_true',
        help='use a gated linear unit for the first linear layer in the MLP')
    parser.add_argument(
        '--enable_backward',
        action='store_true',
        help='Whether to perform backward pass (compute gradients)')

    parser.set_defaults(add_bias_linear=False)
    parser.set_defaults(gated_linear_unit=False)
    parser.set_defaults(enable_backward=False)
    args, rest_args = parser.parse_known_args()

    transformer_config = get_config()
    input_, state_dict = get_init_params(args, transformer_config)
    net = TestModel(transformer_config, args.input_size)
    ms.load_param_into_net(net, state_dict)

    if not args.enable_backward:
        if args.add_bias_linear:
            output, bias = net(input_)
            assert bias is not None
        else:
            output = net(input_)
    else:
        weights = ParameterTuple(net.trainable_params())
        train_network = nn.ForwardValueAndGrad(net, weights=weights, get_all=True, get_by_list=True)
        outputs, grads = train_network(input_)
        if args.add_bias_linear:
            output, bias = outputs
            assert bias is not None
        else:
            output = outputs
        grads_npu = to_numpy_list(grads)
        standard = DoubleBenchmarkStandard(dtype="bfloat16")
        grads_gpu = get_grads('gpu')
        grads_golden = get_grads('cpu')
        for grad_npu, grad_gpu, grad_golden in zip(grads_npu, grads_gpu, grads_golden):
            DoubleBenchmarkComparator.check_pass_or_not(grad_npu, grad_gpu, grad_golden, standard)
    output_npu = output.asnumpy()
    standard = DoubleBenchmarkStandard(dtype="bfloat16")
    output_gpu = get_gpu_datas(args)
    output_golden = get_golden_datas(args)
    DoubleBenchmarkComparator.check_pass_or_not(output_npu, output_gpu, output_golden, standard)

    print(f"Test Case Finished: input_size:{args.input_size}, add_bias-linear:{args.add_bias_linear}, "
          f"gated_linear_unit:{args.gated_linear_unit}, enable_backward:{args.enable_backward}.")
