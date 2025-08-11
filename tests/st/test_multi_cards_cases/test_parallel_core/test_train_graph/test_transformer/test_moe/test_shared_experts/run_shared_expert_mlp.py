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
"""Test module for testing SharedExpertMLP used for mindformers."""
import argparse
import os

import mindspore as ms
from mindspore import nn
from mindspore.communication import init
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.moe.shared_experts import SharedExpertMLP, MLPSubmodules
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.parallel_core.training_graph.device_matrix import layout
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
from tests.st.test_ut.test_parallel_core.test_training_graph.test_transformer.test_moe.test_shared_experts.data_gen_utils import get_init_params, get_golden_datas, get_gpu_datas
from tests.utils.double_benchmark import DoubleBenchmarkComparator, DoubleBenchmarkStandard

ms.context.set_context(deterministic="ON")
ms.set_context(mode=ms.GRAPH_MODE)


def get_config():
    """get TransformerConfig for test"""
    return TransformerConfig(tensor_model_parallel_size=args.tp,
                             num_layers=1,
                             num_attention_heads=2,
                             hidden_size=16,
                             ffn_hidden_size=16,
                             moe_shared_expert_intermediate_size=16,
                             hidden_act="silu",
                             add_bias_linear=False,
                             compute_dtype='bfloat16',
                             params_dtype='float32',
                             moe_router_dtype='float32',
                             )


class TestModel(nn.Cell):
    """Model for test"""

    def __init__(self, config: TransformerConfig):
        super(TestModel, self).__init__()
        self.config = config
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        if self.rank_id is not None:
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True,
                # device_num=self.worker_num # device_num is often set by environment
            )
            init()  # Initialize communication
            self.tp = self.config.tensor_model_parallel_size \
                if self.config.tensor_model_parallel_size is not None else 1
            self.dp = self.config.data_parallel_size if self.config.data_parallel_size is not None else 1
            self.pp = self.config.pipeline_model_parallel_size \
                if self.config.pipeline_model_parallel_size is not None else 1
            initialize_model_parallel(tensor_model_parallel_size=self.tp, data_parallel_size=self.dp,
                                      pipeline_model_parallel_size=self.pp)

        layout.init_layout(self.config)
        self.mlp = SharedExpertMLP(config=config, submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear,
                                                                           linear_fc2=RowParallelLinear))

    def construct(self, hidden_states):
        """This avoids graph compilation errors due to unsupported return types."""
        mlp_output = self.mlp(hidden_states)
        return mlp_output[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tp',
        default=1,
        required=False,
        type=int,
        help='tensor_parallel')
    parser.add_argument(
        '--gate',
        action='store_true',
        help='use a gated linear unit in the SharedExpertMLP')

    parser.set_defaults(gate=False)
    args, rest_args = parser.parse_known_args()

    transformer_config = get_config()
    transformer_config.use_shared_expert_gating = args.gate
    input_, state_dict = get_init_params(transformer_config)
    net = TestModel(transformer_config)
    ms.load_param_into_net(net, state_dict)

    output = net(input_)
    output_npu = output.asnumpy()
    standard = DoubleBenchmarkStandard(dtype="bfloat16")
    output_gpu = get_gpu_datas(args)
    output_golden = get_golden_datas(args)
    DoubleBenchmarkComparator.check_pass_or_not(output_npu, output_gpu, output_golden, standard)
