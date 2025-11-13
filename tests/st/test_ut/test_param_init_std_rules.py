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
"""test param_init_std_rules api."""

import os
import pytest
import numpy as np
import mindspore as ms
from mindspore.communication import init

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.training_graph.transformer.transformer_layer import TransformerLayerSubmodules, \
    TransformerLayer
from mindformers.parallel_core.training_graph.transformer.norm import Norm
from mindformers.parallel_core.training_graph.transformer.attention import SelfAttention, \
    SelfAttentionSubmodules
from mindformers.parallel_core.training_graph.base_models.gpt.gpt_model import GPTModel
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.parallel_core.training_graph.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.parallel_core.training_graph.transformer.flash_attention import FlashAttention
from mindformers.core.context.build_context import build_context

class GPTModelNet:
    """A simple net for test"""

    def __init__(self):

        # Model dimensions
        self.hidden_size = 4096
        self.seq_length = 16
        self.batch_size = 16
        self.ffn_hidden_size = 64
        self.num_attention_heads = 4
        self.num_layers = 4

        self.compute_dtype = ms.bfloat16
        self.param_init_dtype = ms.float32

        # Parallelism
        self.tensor_parallel = 1
        self.rank_id = None
        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None

        if self.rank_id is not None:
            ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True)
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
            compute_dtype='bfloat16',
            layernorm_compute_dtype='float32',
            normalization="LayerNorm",
            num_layers=self.num_layers,
            params_dtype='float32',
            param_init_std_rules = [{"target": ".*linear_proj.weight.*", "init_method_std": 1}],
        )

        # Submodules
        submodules = TransformerLayerSubmodules(
            input_layernorm=Norm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=FlashAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
        )
        self.submodules_spec = ModuleSpec(module=TransformerLayer, submodules=submodules)

    def build_model(self):
        """Build and initialize gpt model"""
        net = GPTModel(
            config=self.config,
            transformer_layer_spec=self.submodules_spec,
            vocab_size=16,
            max_sequence_length=32,
            position_embedding_type='rope',
        )

        return net


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_param_init_std_rules():
    """
    Feature: get_optimizer_grouped_parameters api
    Description: Test get_optimizer_grouped_parameters function
    Expectation: No exception.
    """
    build_context({"use_legacy": False})
    ms.set_context(mode=1, device_target='CPU')  # GRAPH_MODE is typical for MindSpore model execution
    ms.set_seed(42)
    np.random.seed(42)
    net = GPTModelNet()
    gpt_model = net.build_model()
    param_target = "linear_proj.weight"
    expected_std = 1

    for param_name, param in gpt_model.parameters_and_names():
        if param_target in param_name:
            assert abs(param.std()-expected_std) < 0.005
