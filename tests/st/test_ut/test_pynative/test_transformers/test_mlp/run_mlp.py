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
"""Run MLP accuracy test with configurable parameters via args"""
import argparse

import mindspore as ms
from mindspore import nn, ParameterTuple
from data_gen_utils import get_init_params, get_golden_datas, get_gpu_datas, get_grads
from tests.utils.double_benchmark import DoubleBenchmarkComparator, DoubleBenchmarkStandard
from tests.utils.tensor_utils import to_numpy_list
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
from mindformers.pynative.layers.linear import Linear
from mindformers.models.utils import convert_mstype


class MLPRunner:
    """Class to manage MLP model and weights"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.input_size = self.args.input_size
        self.hidden_act = self.args.hidden_act
        self.add_bias_linear = self.args.add_bias_linear
        self.gated_linear_unit = self.args.gated_linear_unit
        self.compute_dtype = self.args.compute_dtype
        self.params_dtype = self.args.params_dtype
        self.enable_backward = self.args.enable_backward

        self.transformer_config = self._get_config()
        self.input_, self.state_dict = get_init_params(self.args, self.transformer_config)

    def _get_config(self):
        """get TransformerConfig for test"""
        return TransformerConfig(
            num_layers=1,
            num_attention_heads=2,
            hidden_size=16,
            ffn_hidden_size=16,
            add_bias_linear=self.add_bias_linear,
            hidden_act=self.hidden_act,
            gated_linear_unit=self.gated_linear_unit,
            compute_dtype=self.compute_dtype,
            params_dtype=self.params_dtype
        )

    def build_model(self):
        """Build and initialize MLP model"""
        net = TestModel(self.transformer_config, self.input_size)
        ms.load_param_into_net(net, self.state_dict)
        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()

        # Validate compute_dtype and params_dtype are correctly passed to linear_fc1
        expected_compute_dtype = convert_mstype(self.compute_dtype)
        expected_params_dtype = convert_mstype(self.params_dtype)
        actual_compute_dtype = net.mlp.linear_fc1.compute_dtype
        actual_params_dtype = net.mlp.linear_fc1.params_dtype

        assert actual_compute_dtype == expected_compute_dtype, (
            f"compute_dtype mismatch: expected {expected_compute_dtype}, "
            f"but got {actual_compute_dtype} in linear_fc1"
        )
        assert actual_params_dtype == expected_params_dtype, (
            f"params_dtype mismatch: expected {expected_params_dtype}, "
            f"but got {actual_params_dtype} in linear_fc1"
        )

        # If compute_dtype is not bfloat16 or params_dtype is not float32,
        # only validate dtype passing and return early
        if self.compute_dtype != 'bfloat16' or self.params_dtype != 'float32':
            print(f"Dtype Validation Test Case Finished: input_size:{self.input_size}, hidden_act:{self.hidden_act}, "
                  f"add_bias_linear:{self.add_bias_linear}, gated_linear_unit:{self.gated_linear_unit}, "
                  f"compute_dtype:{self.compute_dtype}, params_dtype:{self.params_dtype}, "
                  f"enable_backward:{self.enable_backward}. "
                  f"Dtype validation passed: compute_dtype and params_dtype are correctly passed to linear_fc1.")
            return

        if not self.enable_backward:
            if self.add_bias_linear:
                output, bias = net(self.input_)
                assert bias is not None
            else:
                output = net(self.input_)
        else:
            weights = ParameterTuple(net.trainable_params())
            train_network = nn.ForwardValueAndGrad(net, weights=weights, get_all=True, get_by_list=True)
            outputs, grads = train_network(self.input_)
            if self.add_bias_linear:
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
        if self.hidden_act is not None:
            standard = DoubleBenchmarkStandard(dtype="bfloat16")
            output_gpu = get_gpu_datas(self.args)
            output_golden = get_golden_datas(self.args)
            DoubleBenchmarkComparator.check_pass_or_not(output_npu, output_gpu, output_golden, standard)
            print(f"Accuracy Test Case Finished: input_size:{self.input_size}, hidden_act:{self.hidden_act}, "
                  f"add_bias_linear:{self.add_bias_linear}, gated_linear_unit:{self.gated_linear_unit}, "
                  f"compute_dtype:{self.compute_dtype}, params_dtype:{self.params_dtype}, "
                  f"enable_backward:{self.enable_backward}.")
        else:
            print(f"Functional Test Case Finished: input_size:{self.input_size}, hidden_act:{self.hidden_act}, "
                  f"add_bias_linear:{self.add_bias_linear}, gated_linear_unit:{self.gated_linear_unit}, "
                  f"compute_dtype:{self.compute_dtype}, params_dtype:{self.params_dtype}, "
                  f"enable_backward:{self.enable_backward}.")


class TestModel(nn.Cell):
    """Model for test"""
    def __init__(self, config: TransformerConfig, input_size=None):
        super().__init__()
        self.config = config
        self.mlp = MLP(submodules=MLPSubmodules(linear_fc1=Linear, linear_fc2=Linear),
                       config=config, input_size=input_size)

    def construct(self, hidden_states):
        """This avoids graph compilation errors due to unsupported return types."""
        if self.config.add_bias_linear:
            mlp_output, mlp_bias, _ = self.mlp(hidden_states)
            return mlp_output, mlp_bias
        mlp_output = self.mlp(hidden_states)
        return mlp_output[0]


def main():
    parser = argparse.ArgumentParser(description="Run MLP test")
    parser.add_argument(
        '--input_size',
        default=None,
        required=False,
        type=int,
        help='hidden_size of linear')
    parser.add_argument(
        '--hidden_act',
        default=None,
        required=False,
        type=str,
        help='activation func used in MLP')
    parser.add_argument(
        '--add_bias_linear',
        action='store_true',
        help='include a bias term in linear layers')
    parser.add_argument(
        '--gated_linear_unit',
        action='store_true',
        help='use a gated linear unit for the first linear layer in MLP')
    parser.add_argument(
        '--compute_dtype',
        default='bfloat16',
        required=False,
        type=str,
        help='compute dtype used in MLP')
    parser.add_argument(
        '--params_dtype',
        default='float32',
        required=False,
        type=str,
        help='params dtype used in MLP')
    parser.add_argument(
        '--enable_backward',
        action='store_true',
        help='Whether to perform backward pass (compute gradients)')

    parser.set_defaults(add_bias_linear=False)
    parser.set_defaults(gated_linear_unit=False)
    parser.set_defaults(enable_backward=False)
    args = parser.parse_args()

    ms.context.set_context(deterministic="ON")
    ms.set_context(mode=ms.PYNATIVE_MODE)

    # Prepare input and run
    runner = MLPRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
