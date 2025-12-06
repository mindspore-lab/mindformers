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
Test module for testing tools check_rules for mindformers.
"""
import pytest
from mindformers.tools.check_rules import (
    _restore_net_type,
    _rule_fa_only_for_train,
    _check_keyword_gen_dataset,
    _check_context_parallel_algo_valid,
    _check_recompute
    )
from mindformers import MindFormerConfig


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_restore_net_type():
    """
    Feature: Check rules
    Description: Test check_rules.
    Expectation: Run successfully.
    """
    config = MindFormerConfig()
    config.set_value('model.model_config.compute_dtype', 'bfloat16')
    config.set_value('model.model_config.param_init_type', 'float32')
    _restore_net_type(config=config)

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_rule_fa_only_train():
    """
    Feature: Check rules
    Description: Test check_rules.
    Expectation: Run successfully.
    """
    config = MindFormerConfig()
    config.set_value('model.model_config.use_flash_attention', True)
    _rule_fa_only_for_train(config=config, mode="train")

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_context_parallel_algo_valid():
    """
    Feature: Check rules
    Description: Test check_rules.
    Expectation: Run successfully.
    """
    config = MindFormerConfig()
    config.set_value('model.model_config.n_kv_heads', None)
    config.set_value('model.model_config.multi_query_group_num', 2)
    config.set_value('model.model_config.num_heads', None)
    config.set_value('model.model_config.num_attention_heads', 32)
    config.set_value('parallel_config.context_parallel_algo.value', "ulysses_cp")
    with pytest.raises(ValueError, match=r"cp \* mp <= attention head"):
        _check_context_parallel_algo_valid(config=config, cp=8, mp=8)

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_keyword_gen_dataset():
    """
    Feature: Check rules
    Description: Test check_rules.
    Expectation: Run successfully.
    """
    config = MindFormerConfig()
    config.set_value('model.model_config.seq_length', 101)
    config.set_value('do_eval', False)
    config.set_value('metric', [{"type": "ADGENMetric"}, {"type": "PerplexityMetric"}])

    # train dataset
    config.set_value('train_dataset.data_loader.type', "ADGenDataLoader")
    config.set_value('train_dataset.max_source_length', 50)
    config.set_value('train_dataset.max_target_length', 50)

    # eval dataset
    config.set_value('eval_dataset.data_loader.type', "ADGenDataLoader")
    config.set_value('eval_dataset.data_loader.phase', "eval")
    config.set_value('eval_dataset.max_source_length', 101)
    config.set_value('eval_dataset.max_target_length', 20)
    config.set_value('eval_dataset_task.dataset_config.data_loader.phase', "eval")

    _check_keyword_gen_dataset(config=config, mode='train')

@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_check_recompute():
    """
    Feature: Check rules
    Description: Test check_rules.
    Expectation: Run successfully.
    """
    config = MindFormerConfig()
    config.set_value("swap_config.swap", True)
    config.set_value("recompute_config.recompute", True)
    config.set_value("recompute_config.select_recompute", True)
    config.set_value("recompute_config.select_comm_recompute", True)
    _check_recompute(config=config)
