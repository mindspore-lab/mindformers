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
"""test infer transformer core utils"""
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.communication._comm_helper import _is_initialized

from research.deepseek3.deepseek3_config import DeepseekV3Config
from research.deepseek3.moe import RoutedParallelMLP, ExpertParallelMoE
from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel


class ExpertParallelMoENet(nn.Cell):
    """A model class of new transform layer."""
    def __init__(self, config: DeepseekV3Config):
        super(ExpertParallelMoENet, self).__init__()
        if _is_initialized():
            initialize_model_parallel(pipeline_model_parallel_size=1,
                                      expert_model_parallel_size=config.parallel_config.expert_parallel,
                                      tensor_model_parallel_size=4,
                                      order='tp-ep-dp-pp')
        ffn = RoutedParallelMLP(config)
        self.routed_experts = ExpertParallelMoE(ffn, config.hidden_size, config.moe_config,
                                                config.parallel_config.use_alltoall)

    def construct(self, hidden_states: Tensor):
        """Construct for Transformer Layer network.

        Args:
            hidden_states (Tensor): Input tensor of shape (1, batch_size * seq_length, hidden_size)
        Returns:
            Tensor: Output hidden states after transformer layer processing
        """
        output = self.routed_experts(hidden_states)
        return output
