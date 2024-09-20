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
"""test configs and initializations"""

import pytest

from mindformers.experimental.parallel_core.pynative.config import (
    init_configs_from_cmd
)

allowed_error = 1e-5


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_config_from_cmd():
    """
    Feature: init configs from command, as the basic test.
    Description: Test to initialize configs from command line
    Expectation: success
    """
    base_cmd = "--seed 123 --lr 1e-2 --tensor-model-parallel-size 1 --num-layers 6 --train-iters 10 \
                --data-path ./dataset --vocab-size 32000 --num-attention-heads 32 \
                 --hidden-size 4096 --ffn-hidden-size 16384"
    model_type = "model_config"
    all_config = init_configs_from_cmd(base_cmd, model_type)
    assert all_config.training_config.seed == 123
    assert abs(all_config.optimizer_config.learning_rate - 1e-2) < allowed_error
    assert all_config.parallel_config.tensor_model_parallel_size == 1
    assert all_config.dataset_config.dataset_dir == "./dataset"
    assert all_config.model_config.vocab_size == 32000
    assert all_config.model_config.num_attention_heads == 32
    assert all_config.model_config.hidden_size == 4096
    assert all_config.model_config.ffn_hidden_size == 16384
